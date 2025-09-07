import os
from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
INDEX_NAME = "diabetes-diagnosis-db"
index = pinecone.Index(INDEX_NAME)

model = SentenceTransformer('all-MiniLM-L6-v2')

query_text = "average glucose levels and insulin"
query_vector = model.encode(query_text).tolist()

results = index.query(
    vector=query_vector,
    top_k=5,
    filter={"dataset": "hupa-ucm"}
)

print(results)
