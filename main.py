from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pdfplumber
import io
import os

from vector_store import process_and_store_document

load_dotenv()

app = FastAPI(
    title="Multi-Agent Insight Generator",
    description="AI Engine for NAAC Compliance Reports",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "Online", "message": "The AI Engine is Live! 🚀"}

@app.post("/upload-pdf/")
async def upload_and_parse_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed Bhai!")

    try:
        file_bytes = await file.read()
        extracted_text = ""
        
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF is empty or unreadable!")

        # --- NEW: Send the text to the Vector DB ---
        # Ye line text ko chunks mein todegi aur Pinecone mein save karegi
        num_chunks = process_and_store_document(extracted_text, file.filename)
        
        return {
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "chunks_created": num_chunks, # Kitne tukde hue wo return kar rahe hain
            "message": "Success! PDF parsed and stored in Pinecone Vector DB! 🚀"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")