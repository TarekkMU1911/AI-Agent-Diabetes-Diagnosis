import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone ,ServerlessSpec
from dotenv import load_dotenv
import torch
device = torch.device("cpu")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("diabetes-index")

# Example query
query = "What are diabetes complications?"
query_vector = model.encode(query).tolist()

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

# Print retrieved results
for match in results["matches"]:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']:.4f}")
    print(f"Text: {match['metadata']['text']}")
    print("="*80)