import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
import pickle
import pdfplumber

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=os.getenv("HF_TOKEN"))

class RAGService:
    def __init__(self, knowledge_base_dir: str = "knowledge_base"):
        """Initialize the RAG service with a knowledge base directory."""
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store_path = "faiss_index"
        self.chunks_file = "document_chunks.pkl"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=os.getenv("HF_TOKEN"))
        self.vector_store = None
        self.document_chunks = []
        
        # Create knowledge base directory if it doesn't exist
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # Initialize or load the vector store
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base by loading existing or creating new index."""
        # Check if we have a saved index and it's up to date
        if os.path.exists(self.vector_store_path) and os.path.exists(self.chunks_file):
            # Load existing index
            self.vector_store = faiss.read_index(self.vector_store_path)
            with open(self.chunks_file, 'rb') as f:
                self.document_chunks = pickle.load(f)
            print(f"Loaded existing knowledge base with {len(self.document_chunks)} chunks")
        else:
            # Process all PDFs in the knowledge base
            self._process_knowledge_base()
    
    def _process_knowledge_base(self):
        """Process all PDFs in the knowledge base directory."""
        print("Processing knowledge base PDFs...")
        pdf_files = [f for f in os.listdir(self.knowledge_base_dir) 
                    if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print("No PDF files found in the knowledge base directory.")
            return
            
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        total_chunks = 0
        for pdf_file in pdf_files:
            try:
                pdf_path = os.path.join(self.knowledge_base_dir, pdf_file)
                print(f"Processing {pdf_file}...")
                success, message = self.process_pdf(pdf_path)
                if success:
                    print(f"  ✓ {message}")
                    total_chunks = len(self.document_chunks)  # Update total chunks
                else:
                    print(f"  ✗ {message}")
            except Exception as e:
                print(f"  Error processing {pdf_file}: {str(e)}")
        
        if total_chunks > 0:
            print(f"Successfully processed {total_chunks} chunks from {len(pdf_files)} PDFs")
            # Save the vector store and chunks
            faiss.write_index(self.vector_store, self.vector_store_path)
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            print(f"Saved vector store and chunks to disk")
    
    def process_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """Process a PDF file and add it to the knowledge base."""
        try:
            # Extract text from PDF
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            
            if not text.strip():
                return False, f"No text extracted from {os.path.basename(pdf_path)}"
            
            # Split text into chunks (simple implementation - can be enhanced)
            chunk_size = 1000  # Adjust based on your needs
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            
            # Generate embeddings for chunks
            chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
            chunk_embeddings = np.array(chunk_embeddings).astype('float32')
            
            # Create or update FAISS index
            if self.vector_store is None:
                dimension = chunk_embeddings.shape[1]
                self.vector_store = faiss.IndexFlatL2(dimension)
                self.vector_store.add(chunk_embeddings)
            else:
                self.vector_store.add(chunk_embeddings)
            
            # Save document chunks
            self.document_chunks.extend(chunks)
            
            return True, f"Processed {len(chunks)} chunks from {os.path.basename(pdf_path)}"
            
        except Exception as e:
            return False, f"Error processing {os.path.basename(pdf_path)}: {str(e)}"
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve the top-k most relevant text chunks for a query."""
        if self.vector_store is None or not self.document_chunks:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding, dtype='float32')
            
            # Search the FAISS index
            distances, indices = self.vector_store.search(query_embedding, top_k)
            
            # Return the corresponding text chunks
            return [self.document_chunks[i] for i in indices[0] if i < len(self.document_chunks)]
            
        except Exception as e:
            print(f"Error retrieving relevant chunks: {str(e)}")
            return []

# Global instance of the RAG service
rag_service = RAGService()
