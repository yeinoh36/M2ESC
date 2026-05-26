import os
import json
import torch
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

ACTION_DB_PATH = "/data1/yioh/code/ai/data/knowledge_db.json"
BM25_INDEXES = {}

_st_model = None
_device = "cpu"

def init_databases():
    global BM25_INDEXES
    if not BM25Okapi: return
    
    raw_databases = {"wikihow": [], "reddit": [], "counselchat": []}
    if os.path.exists(ACTION_DB_PATH):
        print(f"📦 Loading Action DB from {ACTION_DB_PATH}...")
        with open(ACTION_DB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw_databases["wikihow"] = data.get("wikihow", [])
            raw_databases["reddit"] = data.get("reddit", [])
            raw_databases["counsel"] = data.get("counsel", []) 
    else:
        print(f"⚠️ Warning: Action DB file not found at {ACTION_DB_PATH}. BM25 indexes will be empty.")

    all_texts = []
    
    for db_name, texts in raw_databases.items():
        if texts:
            print(f"✨ Building BM25 Index for [{db_name}] ({len(texts)} docs)...")
            tokenized_corpus = [doc.lower().split() for doc in texts]
            BM25_INDEXES[db_name] = {"bm25": BM25Okapi(tokenized_corpus), "texts": texts}
            all_texts.extend(texts)
        else:
            BM25_INDEXES[db_name] = None
            
    print("✅ BM25 Indexes initialized for all databases.")

def init_st_model(gpu_id='cpu'):
    global _st_model, _device

    _device = f"cuda:{gpu_id}" if gpu_id != 'cpu' else "cpu"

    print(f"Initializing SentenceTransformer model (BAAI/bge-m3) on {_device}...")
    _st_model = SentenceTransformer('BAAI/bge-m3', device=_device)
    print("✅ SentenceTransformer model loaded successfully.")

def perform_hybrid_search(target_db, keywords, search_query, bm25_top_k=10, final_top_k=3):
    if not search_query or not search_query.strip():
        return {"error": "Search query is empty. Please provide a valid search query."}

    target_db = target_db.lower()
    if target_db not in BM25_INDEXES or not BM25_INDEXES.get(target_db):
        return {"error": f"No BM25 index available for target database '{target_db}'. Please check the database name and ensure it is initialized."}

    db_info = BM25_INDEXES.get(target_db)
    if not db_info:
        return {"error": f"BM25 index for '{target_db}' is not initialized. Please initialize the databases first."}

    bm25_model = db_info["bm25"]
    target_texts = db_info["texts"]

    # =======================================================
    # 💡 1단계: 프롬프트 의도대로 'keywords'를 이용한 BM25 고속 검색
    # =======================================================
    if isinstance(keywords, list):
        keyword_str = " ".join(keywords)
    else:
        keyword_str = str(keywords)
        
    tokenized_query = keyword_str.lower().split()
    candidates = bm25_model.get_top_n(tokenized_query, target_texts, n=min(bm25_top_k, len(target_texts)))
    
    if not candidates:
        return {"error": "No candidates retrieved from BM25 search. Please check the keywords and target database."}

    # =======================================================
    # 💡 2단계: 프롬프트 의도대로 'search_query(문단)'을 이용한 임베딩 비교
    # =======================================================
    query_emb = _st_model.encode(search_query, convert_to_tensor=True, device=_st_model.device, show_progress_bar=False)
    candi_emb = _st_model.encode(candidates, convert_to_tensor=True, device=_st_model.device, show_progress_bar=False)

    cos_scores = util.pytorch_cos_sim(query_emb, candi_emb)[0]
    
    top_results_k = min(final_top_k, len(candidates))
    top_results = torch.topk(cos_scores, k=top_results_k)

    results_str = []
    for idx, score in zip(top_results[1], top_results[0]):
        doc_text = candidates[idx.item()]
        clean_text = doc_text.replace('\n', ' / ') 
        results_str.append(f"- [Score: {score.item():.2f}] {clean_text}")

    return "\n".join(results_str)
# ==========================================