import argparse, os, json, time, torch
from vllm import LLM, SamplingParams
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

from src.memory import manage_dialog_context # #1 Memory Architect
from src.router import predict_esc_stage_and_strategy # #2 Strategic Router
from src.specialized import detect_information_gap, analyze_emotion_trajectory, generate_grounded_solution # #3 Expert Ensemble
from src.tag import init_tag_models
from src.rag import init_databases, init_st_model
from src.respond import generate_response # #4 Synthesis Generator

MODEL_PATH = "/data1/llm-models/Qwen3-14B"

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    primary_gpu = 0 if torch.cuda.is_available() else 'cpu'

    # 모델 초기화
    init_tag_models(gpu_id=primary_gpu)
    init_st_model(gpu_id=primary_gpu)
    init_databases()
    
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=len(args.gpu_ids.split(',')), gpu_memory_utilization=0.5)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0.0)

    data_path = '/data1/yioh/MM/data/ESConv_preprocessed.json'
    os.makedirs('/data1/yioh/MM/results', exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Debug mode: 앞의 3개만 실행
    if args.debug:
        print(f"\n🐛 DEBUG MODE: 앞의 3개 데이터만 실행합니다")
        dataset = dataset[:3]
    
    # Output path 생성
    output_base = f'/data1/yioh/MM/results/ESConv_Ours_results'
    output_path = f"{output_base}{'_debug' if args.debug else ''}.json"

    # ========================================================
    # 1. 상태 추적용 데이터 구조 초기화
    # ========================================================
    active_dialogs = []
    for item in dataset:
        dialog = item.get('dialog', [])
        
        active_dialogs.append({
            "original_dialog_index": item.get("original_dialog_index"),
            "problem_type": item.get("problem_type", ""),
            "full_dialog": dialog,
            "turn_pointer": 0,             # 항상 User 발화 인덱스를 가리킴
            "current_prehistory": "",      # 과거 대화 요약 누적
            "current_history": [],         # 최근 대화 내역 (GT 누적 용도)
            "predictions_list": [],        # 피드백이 있는 결과만 저장될 리스트
            "is_done": False
        })
    
    total_start_time = time.time()
    turn_step = 1

    # ========================================================
    # 2. 계층별 동기화 배치 루프 (Teacher-Forcing 순차 진행)
    # ========================================================
    while True:
        batch_pipeline_data = []
        batch_mapping = []

        for d in active_dialogs:
            if d["is_done"]: continue

            u_idx = d["turn_pointer"]
            
            if u_idx >= len(d["full_dialog"]) - 1:
                d["is_done"] = True
                continue
                
            user_turn = d["full_dialog"][u_idx]
            if user_turn["speaker"] != "user":
                d["is_done"] = True
                continue

            # (1) 이번 턴의 User 대화를 컨텍스트에 추가
            d["current_history"].append(user_turn.copy())

            # (2) Target (Assistant 대화 - 내가 예측해야 할 정답) 확인
            a_idx = u_idx + 1
            target_turn = d["full_dialog"][a_idx]
            
            if target_turn["speaker"] != "assistant":
                d["is_done"] = True
                continue

            # (3) 피드백 확인: Assistant 턴의 annotation에서 feedback 읽기
            has_feedback = False
            feedback_val = ""
            next_u_idx = a_idx + 1
            ann = target_turn.get("annotation", {})
            if 'feedback' in ann and ann['feedback'] is not None:
                has_feedback = True
                feedback_val = ann['feedback']

            p_data = {
                "data": {
                    "prehistory_summary": d["current_prehistory"],
                    "recent_dialog": d["current_history"][:] 
                },
                "result": {
                    "turn_index": u_idx,
                    "target_has_feedback": has_feedback, 
                    "prehistory": d["current_prehistory"],
                    "ground_truth": {
                        "strategy": target_turn.get("annotation", {}).get("strategy", ""),
                        "content": target_turn.get("content", ""),
                        "feedback": feedback_val
                    },
                    "output": {}
                },
                "skip": False
            }
            
            d["_target_turn_backup"] = target_turn 
            d["_next_u_idx_backup"] = next_u_idx
            
            batch_pipeline_data.append(p_data)
            batch_mapping.append(d)

        if not batch_pipeline_data:
            break 

        print(f"\n[{turn_step}턴 차례] 진행 중 (활성 대화: {len(batch_pipeline_data)}개)")
        
        # ----------------------------------------
        # 파이프라인 수행 (Stage 1 ~ 4 모델 순차 생성)
        # ----------------------------------------
        
        # Stage 1: Memory Architect (요약)
        summary_targets = []
        summary_indices = []
        for idx, p in enumerate(batch_pipeline_data):
            recent_dialog = p["data"]["recent_dialog"]
            if len(recent_dialog) > 5:
                summary_targets.append({
                    "prehistory_summary": p["data"]["prehistory_summary"],
                    "older_dialogs": recent_dialog[:-5]
                })
                summary_indices.append(idx)
                p["data"]["recent_dialog"] = recent_dialog[-5:] # 최신 5개로 절삭

        if summary_targets:
            print(f" ⏳ [Stage 1] Memory Architect 작동 중 ({len(summary_targets)}개 대화 요약)")
            try:
                updated_contexts = manage_dialog_context(llm, sampling_params, summary_targets)
                for s_idx, context_json in zip(summary_indices, updated_contexts):
                    if "error" not in context_json:
                        updated_summ = context_json.get("updated_summary", "")
                        batch_pipeline_data[s_idx]["data"]["prehistory_summary"] = updated_summ
                        batch_pipeline_data[s_idx]["result"]["output"]["prehistory_summary"] = updated_summ
                    else:
                        print(f"Memory Architect 에러: {context_json['error']}")
                        batch_pipeline_data[s_idx]["result"]["output"]["error"] = f"Memory Agent failed: {context_json['error']}"
                        batch_pipeline_data[s_idx]["result"]["output"]["raw_output"] = context_json.get("raw_output", "")
                        batch_pipeline_data[s_idx]["skip"] = True
            except Exception as e: 
                print(f"Memory Architect 배치 예측 중 오류 발생: {e}")
                pass

        # Stage 2: Strategic Router
        valid_indices = [i for i, p in enumerate(batch_pipeline_data) if not p["skip"]]
        if valid_indices:
            print(f" ⏳ [Stage 2] Strategic Router 작동 중")
            try:
                router_targets = [batch_pipeline_data[i]["data"] for i in valid_indices]
                routing_results = predict_esc_stage_and_strategy(llm, sampling_params, router_targets, args.strategy_mode)
                for idx, route_info in zip(valid_indices, routing_results):
                    if "error" not in route_info:
                        batch_pipeline_data[idx]["data"]["esc_stage"] = route_info.get("stage", "comforting").lower()
                        batch_pipeline_data[idx]["data"]["predicted_strategies"] = route_info.get("strategies", [])
                        batch_pipeline_data[idx]["result"]["output"]["routing"] = route_info
                    else:
                        print(f"Strategic Router 에러: {route_info['error']}")
                        batch_pipeline_data[idx]["result"]["output"]["error"] = f"Router Agent failed: {route_info['error']}"
                        batch_pipeline_data[idx]["result"]["output"]["raw_output"] = route_info.get("raw_output", "")
                        batch_pipeline_data[idx]["skip"] = True
            except Exception as e: 
                print(f"Strategic Router 배치 예측 중 오류 발생: {e}")
                pass

        # Stage 3: Expert Ensemble
        exploration_idx, comforting_idx, action_idx = [], [], []
        for i, p in enumerate(batch_pipeline_data):
            if p["skip"]: continue
            stage = p["data"].get("esc_stage", "comforting")
            if "exploration" in stage: exploration_idx.append(i)
            elif "action" in stage: action_idx.append(i)
            else: comforting_idx.append(i)

        print(f" ⏳ [Stage 3] Expert Ensemble 작동 중 (Exploration: {len(exploration_idx)}, Comforting: {len(comforting_idx)}, Action: {len(action_idx)})")
        if exploration_idx:
            try:
                gap_res = detect_information_gap(llm, sampling_params, [batch_pipeline_data[i]["data"] for i in exploration_idx])
                for idx, r in zip(exploration_idx, gap_res):
                    if "error" not in r:
                        batch_pipeline_data[idx]["data"]["expert_insights"] = r
                        batch_pipeline_data[idx]["result"]["output"]["expert_insights"] = {"agent": "Exploration", "data": r}
                    else:
                        print(f"Exploration Agent 에러: {r['error']}")
                        batch_pipeline_data[idx]["result"]["output"]["error"] = f"Exploration Agent failed: {r['error']}"
                        batch_pipeline_data[idx]["result"]["output"]["raw_output"] = r.get("raw_output", "")
                        batch_pipeline_data[idx]["skip"] = True
            except Exception as e: 
                print(f"Exploration Agent 배치 예측 중 오류 발생: {e}")
                pass
                
        if comforting_idx:
            try:
                traj_res = analyze_emotion_trajectory(llm, sampling_params, [batch_pipeline_data[i]["data"] for i in comforting_idx])
                for idx, r in zip(comforting_idx, traj_res):
                    if "error" not in r:
                        batch_pipeline_data[idx]["data"]["expert_insights"] = r
                        batch_pipeline_data[idx]["result"]["output"]["expert_insights"] = {"agent": "Comforting", "data": r}
                    else:
                        print(f"Comforting Agent 에러: {r['error']}")
                        batch_pipeline_data[idx]["result"]["output"]["error"] = f"Comforting Agent failed: {r['error']}"
                        batch_pipeline_data[idx]["result"]["output"]["raw_output"] = r.get("raw_output", "")
                        batch_pipeline_data[idx]["skip"] = True
            except Exception as e: 
                print(f"Comforting Agent 배치 예측 중 오류 발생: {e}")
                pass
                
        if action_idx:
            try:
                act_res = generate_grounded_solution(llm, sampling_params, [batch_pipeline_data[i]["data"] for i in action_idx])
                for idx, r in zip(action_idx, act_res):
                    if "error" not in r:
                        batch_pipeline_data[idx]["data"]["expert_insights"] = r
                        batch_pipeline_data[idx]["result"]["output"]["expert_insights"] = {"agent": "Action", "data": r}
                    else:
                        print(f"Action Agent 에러: {r['error']}")
                        batch_pipeline_data[idx]["result"]["output"]["error"] = f"Action Agent failed: {r['error']}"
                        batch_pipeline_data[idx]["result"]["output"]["raw_output"] = r.get("raw_output", "")
                        batch_pipeline_data[idx]["skip"] = True
            except Exception as e: 
                print(f"Action Agent 배치 예측 중 오류 발생: {e}")
                pass

        # Stage 4: Synthesis Generator
        valid_indices = [i for i, p in enumerate(batch_pipeline_data) if not p["skip"]]
        if valid_indices:
            print(f" ⏳ [Stage 4] Synthesis Generator 작동 중 ({len(valid_indices)}개 응답 생성)")
            try:
                responses = generate_response(llm, sampling_params, [batch_pipeline_data[i]["data"] for i in valid_indices])
                for idx, resp in zip(valid_indices, responses):
                    if "error" not in resp:
                        batch_pipeline_data[idx]["result"]["output"]["generated_response"] = resp.get("content", "")
                    else:
                        print(f"Synthesis Generator 에러: {resp['error']}")
                        batch_pipeline_data[idx]["result"]["output"]["error"] = f"Synthesis Generator failed: {resp['error']}"
                        batch_pipeline_data[idx]["result"]["output"]["raw_output"] = resp.get("raw_output", "")
                        batch_pipeline_data[idx]["skip"] = True
            except Exception as e: 
                print(f"Synthesis Generator 배치 예측 중 오류 발생: {e}")
                pass

        # ----------------------------------------
        # 💡 결과 반영 및 GT-Forcing 동기화 
        # ----------------------------------------
        for p_data, d_state in zip(batch_pipeline_data, batch_mapping):
            res = p_data["result"]

            # 1. 갱신된 요약 내용을 기록하여 다음 턴부터 유지
            d_state["current_prehistory"] = p_data["data"]["prehistory_summary"]

            # 2. 피드백이 있는 턴에 한해서만, (에러가 나서 빈값이어도) 기록을 남김
            if res["target_has_feedback"]:
                d_state["predictions_list"].append(res)
            
            # 3. Memory Architect 때문에 잘렸던 히스토리를 현재 대화 상태로 동기화
            d_state["current_history"] = p_data["data"]["recent_dialog"][:]

            # 4. 다음 판을 위해 GT(원본 정답 Assistant 답변)를 히스토리에 삽입!
            d_state["current_history"].append(d_state["_target_turn_backup"].copy())

            # 5. 다음 User 발화 위치로 포인터 이동
            d_state["turn_pointer"] = d_state["_next_u_idx_backup"]

        turn_step += 1

    # ========================================================
    # 3. 최종 결과물 포맷팅 및 디스크 저장
    # ========================================================
    final_results = []
    total_saved_predictions = 0

    for d in active_dialogs:
        # 평가 기록(predictions_list)이 존재하는 대화 블록만 추출
        if d["predictions_list"]:
            total_saved_predictions += len(d["predictions_list"])
            final_results.append({
                "original_dialog_index": d["original_dialog_index"],
                "predictions": d["predictions_list"]
            })

    print("\n" + "="*50)
    print("📊 최종 결과 요약")
    print(f" -> 수집된 피드백 턴 결과물 개수: {total_saved_predictions}개")
    print("="*50)

    print(f"Saving results to {output_path}...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    total_time = time.time() - total_start_time
    print(f"Done. 전체 소요 시간: {total_time / 60:.2f}분")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dialog input and run generation pipeline.")
    parser.add_argument('--strategy-mode', type=str, default='2', help="Strategy prediction mode (1: single-strategy, 2: multi-strategies)")
    parser.add_argument('--gpu-ids', type=str, default='0', help="Comma-separated GPU IDs to use (e.g., '0,1' for multi-GPU)")
    parser.add_argument('--debug', action='store_true', help="Debug mode: run only first 3 samples")
    args = parser.parse_args()
    
    main(args)