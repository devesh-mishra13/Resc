from fastapi import FastAPI, HTTPException
from main import processing_with_Custom_QE
from fastapi.middleware.cors import CORSMiddleware
from llama_index import StorageContext,load_index_from_storage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

storage_context = StorageContext.from_defaults(persist_dir='D:/Resc/api/data/Emed2')
index = load_index_from_storage(storage_context)
print("x")
print(index)

@app.post("/summarizer")
def summarizer(data: dict):
    query = data.get('query')
    n_words = data.get('n_words')
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is missing")
    
    response = processing_with_Custom_QE(query,index,n_words)
    print(response)
    return response
