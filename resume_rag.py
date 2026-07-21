"""
RAG Pipeline over a Resume — built from scratch (no LangChain yet)
Uses Google Gemini for both embeddings and generation.
"""

import os
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

# ============================================================
# STEP 0: SETUP
# ============================================================
load_dotenv()  # reads variables from a local .env file
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")
genai.configure(api_key=api_key)

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-1.5-flash"


# ============================================================
# STEP 1: LOAD THE DOCUMENT
# ============================================================
def load_resume_text():
    """
    In a real pipeline, you'd extract this from the PDF using a library
    like pypdf or pdfplumber. For now, we're using the text directly
    so you can focus on the RAG logic first.
    """
    with open("resume.txt", "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# STEP 2: CHUNK THE DOCUMENT
# ============================================================
def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into overlapping chunks of `chunk_size` characters.
    Overlap ensures we don't cut a sentence/idea awkwardly in half
    between two chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap  # move forward, but overlap a bit
    return [c for c in chunks if c]  # drop empty chunks


# ============================================================
# STEP 3: EMBED THE CHUNKS
# ============================================================
def embed_text(text):
    """
    Converts a piece of text into an embedding vector using Gemini's
    embedding model. Returns a list of floats.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"  # tells the model this is content to be searched later
    )
    return result["embedding"]


def embed_query(text):
    """
    Same idea, but for the USER'S QUESTION.
    Gemini has a separate task_type for queries vs documents —
    this small distinction actually improves retrieval accuracy.
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]


# ============================================================
# STEP 4: SIMILARITY SEARCH (cosine similarity)
# ============================================================
def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=3):
    """
    Compares the query embedding against every chunk embedding,
    and returns the top-k most similar chunks.
    """
    similarities = [
        cosine_similarity(query_embedding, chunk_emb)
        for chunk_emb in chunk_embeddings
    ]
    # get indices of the top-k highest similarity scores
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [chunks[i] for i in top_k_indices]


# ============================================================
# STEP 5: GENERATE THE ANSWER USING RETRIEVED CONTEXT
# ============================================================
def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are answering questions about a person's resume.
Use ONLY the context below to answer. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ============================================================
# STEP 6: PUT IT ALL TOGETHER
# ============================================================
def main():
    print("Loading and chunking resume...")
    text = load_resume_text()
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.\n")

    print("Embedding chunks (this calls the API once per chunk)...")
    chunk_embeddings = [embed_text(chunk) for chunk in chunks]
    print("Done embedding.\n")

    while True:
        question = input("Ask a question about the resume (or 'quit'): ")
        if question.lower() == "quit":
            break

        query_embedding = embed_query(question)
        top_chunks = retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=3)

        answer = generate_answer(question, top_chunks)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()