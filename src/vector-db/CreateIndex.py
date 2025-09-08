import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

pc.create_index(
    name="diabetes-diagnosis-db",
    dimension=768,  
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",       
        region="us-east-1" 
    )
)


