from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import json
import os
import asyncio
import uvicorn

from prompt_opt.semantic_similarity.align_score import AlignScore

CACHE_FILE = "similarity_cache.json"
cache = {}
alignscore_cs = None
executor = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cache
    # Load cache from file if available
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            
    global alignscore_cs
    alignscore_cs = AlignScore(model='xlm-roberta-large', batch_size=32, device='cuda', ckpt_path='/mnt/data/factcheck/Alignscore-models/cz_model/checkpoint_all_final.ckpt', evaluation_mode='nli_sp')

    yield
    # Save cache to file on shutdown
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

app = FastAPI(lifespan=lifespan, debug=True)

class SimilarityRequest(BaseModel):
    pairs: list[tuple[str, str]]

async def compute_similarity_tfidf(pair):
    text1, text2 = pair
    pair_key = hashlib.md5(json.dumps(("tfidf", text1, text2)).encode()).hexdigest()
    print("pair_key", pair_key)
    if pair_key in cache:
        return cache[pair_key]
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    cache[pair_key] = similarity
    return similarity

async def compute_similarity_alignscore_cs_all(pairs):
    all_pair_keys = []
    pair_keys = []
    contexts = []
    claims = []
    for text1, text2 in pairs:
        pair_key = hashlib.md5(json.dumps(("alignscore_cs", text1, text2)).encode()).hexdigest()
        all_pair_keys.append(pair_key)
        if pair_key not in cache:
            pair_keys.append(pair_key)    
            contexts.append(text1)    
            claims.append(text2)
    
    if len(pair_keys) > 0:       
        sims = alignscore_cs.score(contexts=contexts, claims=claims)
        assert len(sims) == len(pair_keys), (len(sims), len(pair_keys))
        for pair_key, sim in zip(pair_keys, sims):
            cache[pair_key] = sim
        
    return [cache[pair_key] for pair_key in all_pair_keys]
    

async def process_request(pairs):
    name2sims = defaultdict(list)
    name2sims["tfidf"] = [await compute_similarity_tfidf(pair) for pair in pairs]
    name2sims["alignscore_cs"] = await compute_similarity_alignscore_cs_all(pairs)
    
    for sims in name2sims.values():
        assert len(sims) == len(pairs)
        
    print("name2sims\n", name2sims)
        
    tasks = []
    for idx in range(len(pairs)):
        e = {}
        for name in name2sims.keys():
            e[name] = name2sims[name][idx]
        tasks.append(e)
            
    print("tasks\n", tasks)

    return tasks


@app.post("/similarity")
async def similarity_endpoint(request: SimilarityRequest):
    try:
        return await asyncio.get_event_loop().run_in_executor(executor, lambda: asyncio.run(process_request(request.pairs)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8444)
