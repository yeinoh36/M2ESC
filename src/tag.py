import torch
import torch.serialization
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig, RobertaTokenizer
from src.emotion.model import PretrainedLMModel
from pytorch_pretrained_bert.optimization import WarmupLinearSchedule

VAD_MODEL_CKPT_PATH = "/data1/yioh/code/ai/maESC/src/emotion/emobank-vad-regression-2520-20.ckpt"
CATEGORICAL_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
GOEMOTIONS_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

_vad_model = None
_vad_tokenizer = None
_cat_model = None
_cat_tokenizer = None
_device = "cpu"

def load_vad_model(model_path, device="cuda"):
    model_name = "roberta-large"
    args = SimpleNamespace(
        task="vad-regression", model="roberta", load_ckeckpoint=True,
        load_dataset="semeval", device=device,
        load_pretrained_lm_weights=True
    )

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    config = RobertaConfig.from_pretrained(model_name)
    config.args = args
    
    # PretrainedLMModel은 src/emotion/model.py에 정의됨
    model = PretrainedLMModel(config, cache_path="./../ckpt/", model_name=model_name).to(device)

    print(f"Loading VAD model weights from: {model_path}")
    torch.serialization.add_safe_globals([WarmupLinearSchedule])
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    print("✅ VAD Model loaded successfully.")

    return model, tokenizer

def load_categorical_model(model_name, device="cuda"):
    print(f"Loading Categorical model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    print("✅ Categorical Model loaded successfully.")

    return model, tokenizer

def init_tag_models(gpu_id='cpu'):
    global _vad_model, _vad_tokenizer, _cat_model, _cat_tokenizer, _device
    
    _device = f"cuda:{gpu_id}" if gpu_id != 'cpu' else "cpu"
        
    print(f"Initializing Emotion Tagging Models on {_device}...")
    _vad_model, _vad_tokenizer = load_vad_model(VAD_MODEL_CKPT_PATH, device=_device)
    _cat_model, _cat_tokenizer = load_categorical_model(CATEGORICAL_MODEL_NAME, device=_device)

def predict_vad(model, tokenizer, text, device):
    if not text: return {"v": 0.0, "a": 0.0, "d": 0.0}
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _, logits = model(**inputs)
        vad_scores = logits[0].cpu().tolist()

    return {"v": vad_scores[0], "a": vad_scores[1], "d": vad_scores[2]}

def predict_categorical(model, tokenizer, text, device):
    if not text: return {}
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu()

    scores = {model.config.id2label[i]: prob.item() for i, prob in enumerate(probs)}
    ordered_scores = OrderedDict()
    for label in GOEMOTIONS_LABELS:
        ordered_scores[label] = scores.get(label, 0.0)
        
    return ordered_scores

def get_emotion_tags(content):
    if _vad_model is None or _cat_model is None:
        print("⚠️ Models not initialized. Call init_tag_models() first.")
        return "None"
        
    if not content:
        return "None"

    # 1. VAD Prediction
    vad = predict_vad(_vad_model, _vad_tokenizer, content, _device)
    v, a, d = vad.get('v', 0), vad.get('a', 0), vad.get('d', 0)
    dim_str = f"Valence: {v:.2f}, Arousal: {a:.2f}, Dominance: {d:.2f}"

    # 2. Categorical Prediction (Top 3 only)
    cat_raw = predict_categorical(_cat_model, _cat_tokenizer, content, _device)
    top_3_emotions = sorted(cat_raw.items(), key=lambda item: item[1], reverse=True)[:3]
    cat_str = ", ".join([f"{k}({v:.2f})" for k, v in top_3_emotions])

    return f"VAD 1-5 [{dim_str}] // Probs 0-1 [{cat_str}]"