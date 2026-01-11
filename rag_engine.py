# rag_engine.py

import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Load Sets chapter data
# --------------------------------------------------
with open("sets.txt", "r", encoding="utf-8") as file:
    topic_text = file.read()

# --------------------------------------------------
# Chunking (safe for math + definitions)
# --------------------------------------------------
def chunk_text_for_sets(text):
    raw_chunks = text.split("\n\n")
    chunks = []

    for chunk in raw_chunks:
        clean_chunk = chunk.strip()
        if len(clean_chunk) > 50:
            chunks.append(clean_chunk)

    return chunks

chunks = chunk_text_for_sets(topic_text)

# --------------------------------------------------
# Embeddings
# --------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)

# --------------------------------------------------
# FAISS Vector Database
# --------------------------------------------------
embedding_dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(chunk_embeddings))

# --------------------------------------------------
# Retrieve relevant chunks
# --------------------------------------------------
def retrieve_chunks(question, k=10):
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    _, indices = index.search(
        np.array(question_embedding),
        k=k
    )

    # Use only top 5 chunks for LLM context
    return [chunks[i] for i in indices[0][:3]]

# --------------------------------------------------
# Phi-3 Explanation (STRICT, DEFINITION-FIRST)
# --------------------------------------------------
def explain_with_phi3(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a strict mathematics teacher for Class 11.

Answer the question ONLY using the information given below.

Rules:
- Use definitions and examples exactly as stated in the content.
- If a mathematical term appears (example: prime number, natural number),
  first identify its definition from the content.
- Do NOT assume or invent conditions.
- Do NOT use outside knowledge.
- If the answer cannot be clearly formed from the content, respond exactly with:
  "Answer not found in the provided material."

Question:
{question}

Content:
{context}

Final Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
    )

    return response.json()["response"]

# --------------------------------------------------
# MCQ Generation (STRICT FORMAT)
# --------------------------------------------------
def generate_mcqs_for_sets(context_chunks, num_questions=4):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a mathematics teacher for Class 11.

Generate EXACTLY {num_questions} multiple-choice questions (MCQs)
based ONLY on the content below.

IMPORTANT RULES:
- Generate EXACTLY {num_questions} questions
- Each question must have 4 options: A, B, C, D
- The answer must be exactly one of A, B, C, or D
- Write the answer line strictly as: Answer: A
- Do NOT add any extra information
- Do NOT repeat questions

Content:
{context}

FORMAT:
Q1. Question
A. option
B. option
C. option
D. option
Answer: A

Q2. Question
A. option
B. option
C. option
D. option
Answer: B

Q3. Question
A. option
B. option
C. option
D. option
Answer: C

Q4. Question
A. option
B. option
C. option
D. option
Answer: D
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
    )

    return response.json()["response"]

# --------------------------------------------------
# Confidence Estimation (Simple & Reliable)
# --------------------------------------------------
def estimate_confidence(context_chunks):
    if len(context_chunks) >= 4:
        return "High", 85
    elif len(context_chunks) >= 2:
        return "Medium", 65
    else:
        return "Low", 40

# --------------------------------------------------
# Public API (Used by Streamlit)
# --------------------------------------------------
def answer_question(question):
    top_chunks = retrieve_chunks(question)

    explanation = explain_with_phi3(question, top_chunks)
    mcqs = generate_mcqs_for_sets(top_chunks, num_questions=4)

    confidence_level, confidence_score = estimate_confidence(top_chunks)

    return explanation, mcqs, confidence_level, confidence_score
