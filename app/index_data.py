import glob, os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage")

def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    coll = client.get_or_create_collection("travel_docs", embedding_function=emb_fn)

    docs, ids, metas = [], [], []
    for path in glob.glob("data/travel_docs/*"):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        docs.append(txt)
        ids.append(path)
        metas.append({"source": path})
    if docs:
        coll.upsert(documents=docs, ids=ids, metadatas=metas)
        print(f"Indexed {len(docs)} docs into Chroma at {CHROMA_DIR}")
    else:
        print("No documents found.")

if __name__ == "__main__":
    main()
