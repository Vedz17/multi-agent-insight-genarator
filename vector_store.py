import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 🚀 NAYA: 'workspace_id' parameter add kiya
def process_and_store_document(text: str, filename: str, workspace_id: str):
    """
    Takes raw text, splits it into chunks, and stores them in a specific Pinecone Namespace.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "naac-report-index"
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100, 
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    documents = text_splitter.create_documents(
        [text], 
        metadatas=[{"source": filename}]
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 🚀 NAYA: 'namespace' set kar diya! Ab data mix nahi hoga.
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        namespace=workspace_id 
    )
    
    return len(chunks)