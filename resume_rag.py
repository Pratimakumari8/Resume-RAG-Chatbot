"""
RAG Pipeline over a Resume — built from scratch (no LangChain yet)
Uses Google Gemini (new google-genai SDK) for both embeddings and generation.
"""

import os
import time
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

# ============================================================
# STEP 0: SETUP
# ============================================================
load_dotenv()  # reads variables from a local .env file
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")

client = genai.Client(api_key=api_key)

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]


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
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values


def embed_query(text):
    """
    Same idea, but for the USER'S QUESTION.
    Gemini has a separate task_type for queries vs documents —
    this small distinction actually improves retrieval accuracy.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result.embeddings[0].values


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

    models_to_try = [GENERATION_MODEL] + FALLBACK_MODELS
    max_retries_per_model = 2
    wait_seconds = 5

    for model_name in models_to_try:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except (ServerError, ClientError) as e:
                is_last_attempt_for_model = attempt == max_retries_per_model - 1
                if not is_last_attempt_for_model:
                    print(f"  ({model_name} busy, retrying in {wait_seconds}s...)")
                    time.sleep(wait_seconds)
                else:
                    print(f"  ({model_name} unavailable, trying next model...)")

    raise RuntimeError("All models are currently unavailable. Try again in a minute.")


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