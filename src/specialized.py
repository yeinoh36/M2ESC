import re
import json
import os
from pathlib import Path
from src.tag import get_emotion_tags
from src.rag import perform_hybrid_search

ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = Path(os.environ.get("M2ESC_PROMPT_DIR", ROOT_DIR / "src" / "prompts"))

def generate_json_from_llm(llm, sampling_params, system_prompt, user_contents):
    tokenizer = llm.get_tokenizer()
    prompts = []
    
    for content in user_contents:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    
    results = []
    for out in outputs:
        raw_output = out.outputs[0].text.strip()
        
        try:
            output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
            start_brace = output.find('{')
            end_brace = output.rfind('}')
            
            if (start_brace == -1 or end_brace == -1):
                results.append({"error": "JSON braces not found", "raw_output": raw_output})
                continue
                
            json_str = output[start_brace : end_brace + 1]
            results.append(json.loads(json_str))
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ LLM 출력 JSON 파싱 실패: {e}")
            results.append({"error": f"JSON parsing failed: {e}", "raw_output": raw_output})
            
    return results


def build_user_content(item):
    prehistory = item.get("prehistory_summary", "")
    recent_dialog = item.get("recent_dialog", [])
    
    dialog_text = "\n".join([f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}" for turn in recent_dialog])
    
    if prehistory and prehistory != "This is the beginning of the conversation.":
        return f"[Long-term Memory Summary]\n{prehistory}\n\n[Recent Dialog]\n{dialog_text}"
    return f"[Recent Dialog]\n{dialog_text}"


# ==========================================
# 3-1. Exploration Agent
# ==========================================
def detect_information_gap(llm, sampling_params, batch_data):
    system_prompt = (PROMPT_DIR / "sp_exploration.txt").read_text(encoding='utf-8')
    user_contents = [build_user_content(item) for item in batch_data]
    return generate_json_from_llm(llm, sampling_params, system_prompt, user_contents)


# ==========================================
# 3-2. Comforting Agent
# ==========================================
def analyze_emotion_trajectory(llm, sampling_params, batch_data):
    system_prompt = (PROMPT_DIR / "sp_comforting.txt").read_text(encoding='utf-8')
    
    user_contents = []
    for item in batch_data:
        prehistory = item.get("prehistory_summary", "")
        recent_dialog = item.get("recent_dialog", [])
        
        tagged_dialog_lines = []
        for turn in recent_dialog:
            speaker = turn.get("speaker", "unknown")
            content = turn.get("content", "")
            
            if speaker.lower() == "user":
                emotion_tag = get_emotion_tags(content)
                tagged_dialog_lines.append(f"{speaker}: {content}\n    ↳ (Emotion Analysis: {emotion_tag})")
            else:
                tagged_dialog_lines.append(f"{speaker}: {content}")
                
        dialog_text = "\n\n".join(tagged_dialog_lines)
        
        if prehistory and prehistory != "This is the beginning of the conversation.":
            user_content = f"[Long-term Memory Summary]\n{prehistory}\n\n[Recent Dialog with Emotion Tags]\n{dialog_text}"
        else:
            user_content = f"[Recent Dialog with Emotion Tags]\n{dialog_text}"
            
        user_contents.append(user_content)
        
    return generate_json_from_llm(llm, sampling_params, system_prompt, user_contents)


# ==========================================
# 3-3. Action Agent
# ==========================================
def generate_grounded_solution(llm, sampling_params, batch_data):
    query_prompt = (PROMPT_DIR / "sp_action_plan.txt").read_text(encoding='utf-8')
    action_prompt = (PROMPT_DIR / "sp_action_act.txt").read_text(encoding='utf-8')
    
    query_contents = [build_user_content(item) for item in batch_data]
    query_results = generate_json_from_llm(llm, sampling_params, query_prompt, query_contents)
    
    final_user_contents = []
    valid_indices = []
    final_results = [None] * len(batch_data)
    for i, (item, query_info) in enumerate(zip(batch_data, query_results)):
        if "error" in query_info:
            final_results[i] = {"error": f"Action Plan JSON parsing failed: {query_info['error']}", "raw_output": query_info.get("raw_output", "")}
            continue

        base_content = build_user_content(item)
        target_db = query_info.get("target_db", "").lower()
        keywords = query_info.get("keywords", [])
        search_query = query_info.get("search_query", "").strip()

        if not target_db or not keywords or not search_query:
            final_results[i] = {"error": "Missing target_db or keywords in Action Plan output.", "raw_output": query_info}
            continue    
        
        retrieved_knowledge = perform_hybrid_search(target_db, keywords, search_query)
        if isinstance(retrieved_knowledge, dict) and "error" in retrieved_knowledge:
            final_results[i] = {"error": f"Retrieval failed: {retrieved_knowledge['error']}"}
            continue
        
        user_content = f"{base_content}\n\n[Retrieved External Knowledge: Top 3]\n{retrieved_knowledge}"
        final_user_contents.append(user_content)
        valid_indices.append(i)
        
    if final_user_contents:
        action_results = generate_json_from_llm(llm, sampling_params, action_prompt, final_user_contents)
        
        for idx, res in zip(valid_indices, action_results):
            final_results[idx] = res

    return final_results