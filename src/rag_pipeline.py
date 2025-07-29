from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Initialize Hugging Face models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Initialize FAISS index
dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
index = faiss.IndexFlatL2(dimension)
chunks = []

def initialize_rag(text_chunks: list[str]):
    """Initialize RAG by embedding text chunks and storing in FAISS."""
    global index, chunks
    if not text_chunks:
        print("Warning: No text chunks provided to initialize_rag")
        chunks = []
        index = faiss.IndexFlatL2(dimension)  # Reset index
        return
    
    chunks = text_chunks
    print(f"Processing {len(chunks)} chunks")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(dimension)  # Reset index
    index.add(np.array(embeddings, dtype=np.float32))

def query_rag(query: str) -> str:
    """Query the RAG pipeline to get an answer."""
    if not chunks or index.ntotal == 0:
        return "No PDF data available. Please upload a PDF first."
    
    # Embed the query
    query_embedding = embedder.encode([query])[0]
    
    # Retrieve top 3 relevant chunks
    k = min(3, len(chunks))
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k=k)
    print(f"FAISS search returned indices: {indices[0]}")
    
    # Safely build context
    context = " ".join([chunks[i] for i in indices[0] if i < len(chunks)])
    if not context:
        return "No relevant context found in the PDF."
    
    # Generate answer using the language model
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    answer = generator(prompt, max_length=500, do_sample=False)[0]["generated_text"]
    print(f"Generated answer: {answer}")  # Debug
    return answer.strip()