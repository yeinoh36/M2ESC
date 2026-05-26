import re
import json
from pathlib import Path

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


def predict_esc_stage_and_strategy(llm, sampling_params, batch_data, strategy_mode="2"):
    if strategy_mode == "1":
        system_prompt = Path("/data1/yioh/code/ai/maESC/src/prompts/sp_router1.txt").read_text(encoding='utf-8')
    else:
        system_prompt = Path("/data1/yioh/code/ai/maESC/src/prompts/sp_router2.txt").read_text(encoding='utf-8')
    # -----------------------------------

    user_contents = []
    
    for item in batch_data:
        prehistory_summary = item.get("prehistory_summary", "")
        recent_dialog = item.get("recent_dialog", [])
        
        dialog_text = "\n".join([f"{turn.get('speaker', 'unknown')}: {turn.get('content', '')}" for turn in recent_dialog])
        
        if prehistory_summary and prehistory_summary != "This is the beginning of the conversation.":
            combined_text = f"[Long-term Memory Summary]\n{prehistory_summary}\n\n[Recent Dialog]\n{dialog_text}"
        else:
            combined_text = f"[Recent Dialog]\n{dialog_text}"

        user_content = f"Please analyze the following conversation context and predict the appropriate ESC stage and strategies:\n\n---\n{combined_text}\n---"
        user_contents.append(user_content)

    json_results = generate_json_from_llm(llm, sampling_params, system_prompt, user_contents)
    
    return json_results