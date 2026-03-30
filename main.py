from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pdfplumber
import io

# 1. Apne custom modules import kar rahe hain
from vector_store import process_and_store_document # PDF save karne ke liye
from agents import app as ai_app # AI se chat karne ke liye

load_dotenv()

# 2. Server Setup
app = FastAPI(
    title="Multi-Agent Insight Generator",
    description="AI Engine for NAAC Compliance Reports",
    version="1.0.0"
)

# 3. CORS Policy (Frontend se connect karne ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 🚪 DOOR 1: THE DATA UPLOAD PIPELINE
# ==========================================
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

        # Text ko Pinecone vector DB mein save karo
        num_chunks = process_and_store_document(extracted_text, file.filename)
        
        return {
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "chunks_created": num_chunks,
            "message": "Success! PDF parsed and stored in Pinecone Vector DB! 🚀"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# ==========================================
# 🚪 DOOR 2: THE AI CHAT PIPELINE (NEW)
# ==========================================

# Data type check karne ke liye schema
class ChatRequest(BaseModel):
    question: str
    domain: str

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    print(f"📩 Naya sawal aaya: {request.question}")
    
    # Notebook banai aur LangGraph AI Engine ko de di
    input_state = {
        "question": request.question,
        "context": "",
        "draft": "",
        "feedback": "",
        "iteration": 0,
        "domain": request.domain
    }
    
    result = ai_app.invoke(input_state)
    
    # Final answer wapas frontend ko bhej diya
    return {"answer": result["draft"]}

# ==========================================
# 🩺 HEALTH CHECK
# ==========================================
@app.get("/")
async def root():
    return {"status": "Online", "message": "The AI Engine is Live! 🚀"}