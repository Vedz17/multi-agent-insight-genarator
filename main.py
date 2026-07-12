from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pdfplumber
import io
import os
import asyncio
from fastapi.responses import StreamingResponse

# Imports from your local files
from vector_store import process_and_store_document
from agents import app as ai_app, report_app, section_app, NAAC_CRITERIA_MAP

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
# DOOR 1: THE DATA UPLOAD PIPELINE
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
            "message": "Success! PDF parsed and stored in private namespace!"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


# ==========================================
# DOOR 2: THE AI CHAT PIPELINE
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
            # Send initial trigger to Frontend UI
            yield "[[STATUS:Researcher is scanning vectors...]]"

            # Use stream_mode="updates" with synchronous node functions
            async for output in ai_app.astream(input_state, stream_mode="updates"):
                for node_name, state_update in output.items():

                    if node_name == "researcher":
                        yield "[[STATUS:Writer is Drafting Response...]]"

                    elif node_name == "writer":
                        yield "[[STATUS:Auditor is validating compliance...]]"

                        # Get the generated draft from the writer node
                        draft = state_update.get("draft", "")

                        if draft:
                            # Stream text in small chunks for the typing effect
                            chunk_size = 15

                            for i in range(0, len(draft), chunk_size):
                                yield draft[i:i + chunk_size]
                                await asyncio.sleep(0.01)

                    elif node_name == "reviewer":
                        yield "[[STATUS:Finalizing Output...]]"

        except Exception as e:
            print(f"Streaming Error: {e}")
            yield f"\n\n[[ERROR: {str(e)}]]"

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==========================================
# DOOR 3: LEGACY NAAC REPORT GENERATOR
# ==========================================
class ReportRequest(BaseModel):
    workspace_id: str
    criterion_id: int
    topics: str = ""


@app.post("/generate-report")
async def generate_naac_report(request: ReportRequest):
    try:
        print(
            f"Initializing Report for Workspace: "
            f"{request.workspace_id}, Criterion: {request.criterion_id}"
        )

        criterion_topics = (
            request.topics
            if request.topics
            else NAAC_CRITERIA_MAP.get(request.criterion_id)
        )

        if not criterion_topics:
            raise HTTPException(
                status_code=400,
                detail="Invalid Criterion ID or missing topics"
            )

        input_state = {
            "workspace_id": request.workspace_id,
            "criterion_id": request.criterion_id,
            "criterion_topics": criterion_topics,
            "context": "",
            "final_report": ""
        }

        final_state = report_app.invoke(input_state)

        return {
            "success": True,
            "content": final_state["final_report"]
        }

    except Exception as e:
        print(f"Report Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# DOOR 4: MULTI-AGENT SECTION GENERATOR
# ==========================================
class SectionRequest(BaseModel):
    workspace_id: str
    criterion_id: int
    section_name: str


@app.post("/generate-section")
async def generate_naac_section(request: SectionRequest):
    """
    Generates one NAAC report section through the bounded
    multi-agent LangGraph pipeline.

    Retriever -> Writer -> Reviewer
                         -> Writer rewrite, maximum once
                         -> Reviewer
                         -> Approved or Flagged
    """

    try:
        print(
            f"Initializing Multi-Agent Section Generation | "
            f"Workspace: {request.workspace_id} | "
            f"Criterion: {request.criterion_id} | "
            f"Section: {request.section_name}"
        )

        if not request.workspace_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Workspace ID is required"
            )

        if not request.section_name.strip():
            raise HTTPException(
                status_code=400,
                detail="Section name is required"
            )

        if request.criterion_id not in NAAC_CRITERIA_MAP:
            raise HTTPException(
                status_code=400,
                detail="Invalid Criterion ID"
            )

        input_state = {
            "workspace_id": request.workspace_id,
            "criterion_id": request.criterion_id,
            "section_name": request.section_name.strip(),
            "context": "",
            "draft": "",
            "feedback": "",
            "review_status": "",
            "iteration": 0
        }

        final_state = await asyncio.to_thread(
            section_app.invoke,
            input_state
        )

        review_status = final_state.get(
            "review_status",
            "unknown"
        )

        print(
            f"Section Generation Complete | "
            f"Section: {request.section_name} | "
            f"Status: {review_status} | "
            f"Writer Attempts: {final_state.get('iteration', 0)}"
        )

        return {
            "success": True,
            "section_name": request.section_name.strip(),
            "content": final_state.get("draft", ""),
            "review_status": review_status,
            "feedback": final_state.get("feedback", ""),
            "writer_attempts": final_state.get("iteration", 0)
        }

    except HTTPException:
        raise

    except Exception as e:
        print(
            f"Multi-Agent Section Generation Error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# HEALTH CHECK
# ==========================================
@app.get("/")
async def root():
    return {
        "status": "Online",
        "message": "The AI Engine is Live!"
    }