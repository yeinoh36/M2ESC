import json
import re
import os
from pathlib import Path

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
            
            if start_brace == -1 or end_brace == -1:
                results.append({"error": "JSON braces not found", "raw_output": raw_output})
                continue
                
            json_str = output[start_brace : end_brace + 1]
            results.append(json.loads(json_str))
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ LLM 출력 JSON 파싱 실패: {e}")
            results.append({"error": f"JSON parsing failed: {e}", "raw_output": raw_output})
            
    return results

def build_synthesis_content(item):
    summary = item.get("prehistory_summary", "No previous history.")
    recent_dialog = item.get("recent_dialog", [])
    dialog_text = "\n".join([f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}" for turn in recent_dialog])

    stage = item.get("esc_stage", "Unknown")
    strategies = item.get("predicted_strategies", [])
    strategies_str = ", ".join(strategies) if isinstance(strategies, list) else str(strategies)
    
    expert_info = item.get("expert_insights", {})
    
    expert_text = ""
    if isinstance(expert_info, dict):
        if "exploration_focus" in expert_info:
            expert_text = (
                f"- Core Stressor: {expert_info.get('core_stressor', 'N/A')}\n"
                f"- Missing Information Gap: {expert_info.get('information_gap', 'N/A')}\n"
                f"- Focus for Next Question: {expert_info.get('exploration_focus', 'N/A')}"
            )
        elif "validation_point" in expert_info:
            expert_text = (
                f"- Emotion Trajectory: {expert_info.get('emotion_trajectory', 'N/A')}\n"
                f"- Current Dominant Emotion: {expert_info.get('current_dominant_emotion', 'N/A')}\n"
                f"- Core Validation Point: {expert_info.get('validation_point', 'N/A')}"
            )
        elif "actionable_advice" in expert_info:
            expert_text = (
                f"- Problem to Solve: {expert_info.get('problem_to_solve', 'N/A')}\n"
                f"- Knowledge Utilized: {expert_info.get('knowledge_utilized', 'N/A')}\n"
                f"- Actionable Advice: {expert_info.get('actionable_advice', 'N/A')}"
            )
        else:
            expert_text = json.dumps(expert_info, indent=2, ensure_ascii=False)
    else:
        expert_text = str(expert_info)

    return (
        f"[Long-term Memory Summary]\n{summary}\n\n"
        f"[Recent Dialog]\n{dialog_text}\n"
        f"[Target ESC Stage]\n{stage}\n\n"
        f"[Target Strategies]\n{strategies_str}\n\n"
        f"[Expert Agent Insights]\n{expert_text}\n\n"
    )

def generate_response(llm, sampling_params, batch_data):
    system_prompt_path = PROMPT_DIR / "sp_respond.txt"
    system_prompt = system_prompt_path.read_text(encoding='utf-8')
    user_contents = [build_synthesis_content(item) for item in batch_data]
    json_results = generate_json_from_llm(llm, sampling_params, system_prompt, user_contents)
    return json_results