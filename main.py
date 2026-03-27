from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pdfplumber
import io
import os

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

# --- NEW: PDF Upload & Parsing Endpoint ---
@app.post("/upload-pdf/")
async def upload_and_parse_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed Bhai!")

    try:
        # Read the file directly into memory (No need to save it to hard drive)
        file_bytes = await file.read()
        
        extracted_text = ""
        
        # Open the PDF from memory using pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            # Loop through all pages and extract text smoothly
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        
        # For now, we are just returning the first 500 characters to verify it worked
        return {
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "preview_text": extracted_text[:500] + "...\n[TEXT TRUNCATED FOR PREVIEW]",
            "message": "PDF successfully parsed! Ready for Vector DB."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")