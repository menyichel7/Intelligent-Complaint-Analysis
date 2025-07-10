import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load  embeddings and IDs
embeddings = np.load('vector_store/embeddings.npy')
ids = np.load('vector_store/ids.npy')

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# FAISS index for searching
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype(np.float32))

def answer_query(query):
    # Generate the query embedding
    query_embedding = model.encode([query])
    
    # Search for similar embeddings
    D, I = index.search(np.array(query_embedding).astype(np.float32), k=5)
    
    # Retrieve relevant chunks and IDs
    relevant_chunks = [ids[i] for i in I[0]]
    answers = [f"Relevant Chunk ID: {chunk}" for chunk in relevant_chunks]
    
    return "Here is the AI-generated answer based on your query.", answers

# Set up Gradio interface
iface = gr.Interface(
    fn=answer_query,
    inputs=gr.inputs.Textbox(label="Ask your question:"),
    outputs=[gr.outputs.Textbox(label="AI Answer"), gr.outputs.Textbox(label="Source Text Chunks")],
    title="Interactive Chat with RAG System",
    description="Type your question and get an answer based on the consumer complaints data."
)

if __name__ == "__main__":
    iface.launch()