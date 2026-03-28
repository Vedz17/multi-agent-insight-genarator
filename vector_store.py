import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def process_and_store_document(text: str, filename: str):
    """
    Takes raw text, splits it into chunks, and stores them in Pinecone.
    """
    # 1. Initialize Pinecone connection
    pc = Pinecone(api_key=os.getenv("pcsk_EizQi_3inc8cwnXEyNqjBpf6rU9iorF19Gk5GauQYZAKA5m33VPaAUUvT9vE9kprWT1sp"))
    index_name = "naac-report-index"
    
    # 2. Setup the Text Splitter (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 3. Create chunks from the raw text
    chunks = text_splitter.split_text(text)
    
    # Optional: Attach metadata (like filename) to each chunk so we know where it came from
    documents = text_splitter.create_documents(
        [text], 
        metadatas=[{"source": filename}]
    )
    
    # 4. Setup Gemini Embeddings (Converting text to numbers)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 5. Push the chunks and embeddings to Pinecone Vector DB
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    
    return len(chunks)
