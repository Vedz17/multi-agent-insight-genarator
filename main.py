from fastapi import FastAPI, UploadFile, File, HTTPException , Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pdfplumber
import io
from fastapi.responses import StreamingResponse

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
async def upload_and_parse_pdf(
    workspaceId: str = Form(...), # 🚀 NAYA: Frontend se ID aayegi
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed Bhai!")

    try:
        file_bytes = await file.read()
        extracted_text = ""
        
        # ... tera pdfplumber wala extract text logic same rahega ...
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="PDF is empty or unreadable!")

        # 🚀 NAYA: process_and_store ko 'workspace_id' pass kar diya
        num_chunks = process_and_store_document(extracted_text, file.filename, workspaceId)
        
        return {
            "filename": file.filename,
            "total_pages": len(pdf.pages),
            "chunks_created": num_chunks,
            "message": "Success! PDF parsed and stored in private namespace! 🚀"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# ==========================================
# 🚪 DOOR 2: THE AI CHAT PIPELINE (STREAMING)
# ==========================================

class ChatRequest(BaseModel):
    question: str
    domain: str
    chat_history: list = []
    workspace_id: str

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    print(f" Naya sawal aaya: {request.question}")
    print(f" Got old messages : {len(request.chat_history)}")
    
    # 1. Ek Generator Function banayenge jo data ko tukdon mein dega
    async def generate_response():
        input_state = {
            "question": request.question,
            "context": "",
            "draft": "",
            "feedback": "",
            "iteration": 0,
            "domain": request.domain,
            "workspace_id": request.workspace_id
        }
        
        try:
            # 2. LangGraph se token-by-token aur Events nikalna
            async for event in ai_app.astream_events(input_state, version="v2"):
                kind = event["event"]
                name = event.get("name", "")

                # 🚀 NINJA TRICK: Jaise hi koi Agent apna kaam shuru kare, UI ko signal bhejo
                if kind == "on_chain_start":
                    name_lower = name.lower()
                    if "research" in name_lower:
                        yield "[[STATUS:🔍 Researcher Agent is searching Pinecone...]]"
                    elif "writer" in name_lower or "draft" in name_lower:
                        yield "[[STATUS:✍️ Writer Agent is drafting the response...]]"
                    elif "review" in name_lower:
                        yield "[[STATUS:🕵️ Reviewer Agent is checking quality...]]"

                # 3. Sirf wahi data pakdo jo AI actual mein type kar raha hai
                elif kind == "on_chat_model_stream":
                    # 🪡 LEAK FIX: Sirf tab stream karo jab ye 'writer' node se aa raha ho
                    if event["metadata"].get("langgraph_node") == "writer":
                        content = event["data"]["chunk"].content
                        if content:
                            yield content
                        
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"\n\n[Error: {str(e)}]"

    # 4. StreamingResponse in tukdon ko 'text/plain' format mein lagataar bhejta rahega
    return StreamingResponse(generate_response(), media_type="text/plain")

# ==========================================
# 🩺 HEALTH CHECK
# ==========================================
@app.get("/")
async def root():
    return {"status": "Online", "message": "The AI Engine is Live! 🚀"}