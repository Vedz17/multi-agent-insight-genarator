from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pdfplumber
import io
import os
from fastapi.responses import StreamingResponse

# Imports from your local files
from vector_store import process_and_store_document 
from agents import app as ai_app, report_app, NAAC_CRITERIA_MAP 

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

# ==========================================
# 🚪 DOOR 1: THE DATA UPLOAD PIPELINE
# ==========================================
@app.post("/upload-pdf/")
async def upload_and_parse_pdf(workspaceId: str = Form(...), file: UploadFile = File(...)):
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
# 🚪 DOOR 2: THE AI CHAT PIPELINE (Streaming Events)
# ==========================================
class ChatRequest(BaseModel):
    question: str
    domain: str
    chat_history: list = []
    workspace_id: str

@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    async def generate_response():
        input_state = {
            "question": request.question,
            "context": "",
            "draft": "",
            "feedback": "",
            "iteration": 0,
            "domain": request.domain,
            "workspace_id": request.workspace_id,
            "chat_history": request.chat_history
        }
        
        try:
            # 🚀 Using native LangGraph events to trigger UI Agents
            async for event in ai_app.astream_events(input_state, version="v2"):
                kind = event["event"]
                name = event.get("name", "")

                # 🟢 Status Trigger Logic (Synchronized with Frontend Keywords)
                if kind == "on_chain_start":
                    name_lower = name.lower()
                    if "researcher" in name_lower:
                        yield "[[STATUS:Researcher is scanning vectors...]]"
                    elif "writer" in name_lower:
                        yield "[[STATUS:Writer is Drafting Response...]]"
                    elif "reviewer" in name_lower:
                        # Frontend expects "Auditor" for the third agent
                        yield "[[STATUS:Auditor is validating compliance...]]"

                # 🔵 Real-time Word-by-word streaming
                elif kind == "on_chat_model_stream":
                    # Only yield chunks from the writer node
                    if event["metadata"].get("langgraph_node") == "writer":
                        content = event["data"]["chunk"].content
                        if content:
                            yield content
                        
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"\n\n[[ERROR: {str(e)}]]"

    return StreamingResponse(
        generate_response(), 
        media_type="text/plain", # Plain text works better for chunked status tags
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" 
        }
    )

# ==========================================
# 🚀 DOOR 3: THE NAAC REPORT GENERATOR
# ==========================================
class ReportRequest(BaseModel):
    workspace_id: str
    criterion_id: int
    topics: str = "" 

@app.post("/generate-report")
async def generate_naac_report(request: ReportRequest):
    try:
        print(f"🚀 Initializing Report for Workspace: {request.workspace_id}, Criterion: {request.criterion_id}")
        
        # Priority: custom topics > dictionary fallback
        criterion_topics = request.topics if request.topics else NAAC_CRITERIA_MAP.get(request.criterion_id)
        
        if not criterion_topics:
            raise HTTPException(status_code=400, detail="Invalid Criterion ID or missing topics")

        input_state = {
            "workspace_id": request.workspace_id,
            "criterion_id": request.criterion_id,
            "criterion_topics": criterion_topics,
            "context": "",
            "final_report": ""
        }

        # Invoking the report_app (Multi-pass logic)
        final_state = report_app.invoke(input_state)
        
        return {
            "success": True,
            "content": final_state["final_report"]
        }
        
    except Exception as e:
        print(f"🚨 Report Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ==========================================
# ✨ DOOR 4: AI REFINEMENT LOGIC
# ==========================================
class RefineRequest(BaseModel):
    current_content: str
    instruction: str
    
# /@app.post("/refine-report")
# async def refine_report(request: RefineRequest):
#     try:
#         from agents import refine_report_logic
#         updated_text = refine_report_logic(
#             request.current_content, 
#             request.instruction
#         )
#         return {"success": True, "content": updated_text}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# 🩺 HEALTH CHECK
# ==========================================
@app.get("/")
async def root():
    return {"status": "Online", "message": "The AI Engine is Live! 🚀"}