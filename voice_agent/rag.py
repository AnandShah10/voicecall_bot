from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embedding(text):
    """Generate embedding vector using OpenAI embeddings."""
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

def add_document(title, content):
    """Add a new company doc with embedding."""
    from .models import CompanyDocument
    embedding = create_embedding(content)
    CompanyDocument.objects.create(title=title, content=content, embedding=embedding)
    print(f"✅ Added: {title}")

def search_similar_docs(query, top_k=2):
    """Find top-k similar docs based on cosine similarity."""
    from .models import CompanyDocument

    query_emb = np.array(create_embedding(query))
    docs = CompanyDocument.objects.all()
    print("Docs................",docs)
    scored_docs = []
    for doc in docs:
        doc_emb = np.array(doc.embedding)
        sim = np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
        scored_docs.append((sim, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc.content for _, doc in scored_docs[:top_k]]
    print(top_docs)
    return top_docs
