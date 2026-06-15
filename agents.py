import os
import time
from typing import TypedDict
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END

load_dotenv()

# =====================================================================
# ⚙️ INITIALIZATIONS (Shared Resources)
# =====================================================================

# --- LLM SETUP (GROQ) ---
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY"),
    max_retries=1
)

# --- PINECONE & EMBEDDINGS ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("naac-report-index")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# --- NAAC CRITERIA MAPPING ---
NAAC_CRITERIA_MAP = {
    1: "Curricular Aspects: Curriculum Planning, Academic Flexibility, Curriculum Enrichment, Feedback System",
    2: "Teaching-Learning and Evaluation: Student Enrollment, Student Diversity, Teaching-Learning Process, Teacher Profile, Evaluation Process",
    3: "Research, Innovations and Extension: Promotion of Research, Resource Mobilization, Innovation Ecosystem, Extension Activities",
    4: "Infrastructure and Learning Resources: Physical Facilities, Library, IT Infrastructure, Maintenance",
    5: "Student Support and Progression: Student Support, Student Progression, Student Participation, Alumni Engagement",
    6: "Governance, Leadership and Management: Institutional Vision, Strategy Development, Faculty Empowerment, Financial Management",
    7: "Institutional Values and Best Practices: Institutional Values, Social Responsibilities, Best Practices, Institutional Distinctiveness"
}

# =====================================================================
# 🟢 PART 1: CHATBOT AGENTS (Optimized for Fast Insights)
# =====================================================================

class GraphState(TypedDict):
    question: str
    context: str
    draft: str
    feedback: str
    iteration: int
    domain: str
    chat_history: list 
    workspace_id: str

def researcher_agent(state: GraphState) -> GraphState:
    """Fetches relevant chunks from Pinecone using workspace namespace."""
    print(f"---🔍 CHAT RESEARCHER: Searching [Namespace: {state.get('workspace_id')}]---")  
    question = state["question"]
    query_vector = embeddings.embed_query(question)
    
    search_results = index.query(
        vector=query_vector, 
        top_k=8, 
        include_metadata=True,
        namespace=state.get("workspace_id") 
    )

    raw_matches = search_results.get("matches", [])
    
    # 🚀 FIX: Lowered threshold from 0.70 to 0.50 to catch basic questions
    valid_chunks = [
        m["metadata"]["text"] 
        for m in raw_matches 
        if m.get("score", 0) >= 0.50 and "text" in m["metadata"]
    ]

    if not valid_chunks:
        print("⚠️ WARNING: Data rejected by Threshold Filter (Score < 0.50).")
        state["context"] = "NO_RELEVANT_DATA"
        return state    

    state["context"] = "\n\n---\n\n".join(valid_chunks)
    return state

def writer_agent(state: GraphState) -> GraphState:
    """Synthesizes an analyst-style response."""
    print(f"---✍️ CHAT WRITER: Drafting Response---")

    if state["context"] == "NO_RELEVANT_DATA":
        state["draft"] = "System Block: Based on the uploaded documents, I could not find relevant institutional data to answer this query. Please ensure valid compliance files are uploaded."
        state["iteration"] = state.get("iteration", 0) + 1 
        return state
    
    question = state["question"]
    context = state["context"]
    
    chat_history_list = state.get("chat_history", [])
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat_history_list])
    
    # 🚀 FIX: Hyper-strict prompt to completely kill hallucinations in Chat
    system_prompt = f"""You are the InsightGen Intelligent Analyst. 
    Provide clear, structured insights based ONLY on the provided institutional documents.

    STRICT ANTI-HALLUCINATION RULES:
    1. STRICT GROUNDING: Use ONLY the context below. Do not use your pre-trained knowledge.
    2. If the context has absolutely nothing to do with the user's query, reply EXACTLY with: "I cannot find specific data regarding this in the uploaded documents."
    3. DO NOT force connections. Do not guess the institution's name unless explicitly written in the context.
    4. Use clean Markdown/bullets and maintain a professional tone.

    DOCUMENTS CONTEXT:
    {context}

    CONVERSATION HISTORY:
    {history_text}

    USER QUERY: {question}
    """
    
    response = llm.invoke(system_prompt)
    state["draft"] = response.content
    state["iteration"] = state.get("iteration", 0) + 1 
    return state

def reviewer_agent(state: GraphState) -> GraphState:
    """Quick check for hallucinations or missing info."""
    print("---🕵️ CHAT REVIEWER: Validating Accuracy---")
    
    # 🚀 FIX: Made the reviewer stricter as well
    system_prompt = f"""Review this response as a strict compliance auditor.
    Does it accurately answer '{state['question']}' using ONLY the provided context?
    Does it hallucinate any names, places, or metrics?
    
    Reply 'PASS' if it is 100% accurate and grounded, or provide critical feedback if it's hallucinating.
    
    DRAFT: {state['draft']}"""
    
    response = llm.invoke(system_prompt)
    state["feedback"] = response.content.strip()
    return state

def review_router(state: GraphState):
    """Controls the loop - Limited to 1 pass for Chat speed."""
    if "PASS" in state.get("feedback", "").upper() or state.get("iteration", 0) >= 1:
        return "end_process"
    return "rewrite_draft"
    
# COMPILING THE CHAT GRAPH
workflow = StateGraph(GraphState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")    
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges("reviewer", review_router, {
    "end_process": END, 
    "rewrite_draft": "writer"
})
app = workflow.compile()


# =====================================================================
# 🚀 PART 2: REPORT GENERATOR AGENTS (Section-by-Section)
# =====================================================================

class ReportGraphState(TypedDict):
    workspace_id: str
    criterion_id: int
    criterion_topics: str
    final_report: str

def report_compiler_loop(state: ReportGraphState) -> ReportGraphState:
    """Iterates through NAAC sub-sections to build a full report."""
    print(f"---📝 SECTION COMPILER: Criterion {state['criterion_id']}---")
    
    full_topics = state['criterion_topics'].split(':')
    criterion_name = full_topics[0]
    sub_sections = [s.strip() for s in full_topics[1].split(',')]
    
    final_combined_report = f"# {criterion_name}\n\n"

    for section in sub_sections:
        print(f"🔍 AI Researcher: Deep scanning for {section}...")
        query_vector = embeddings.embed_query(section)
        
        search_results = index.query(
            vector=query_vector, 
            top_k=6, 
            include_metadata=True, 
            namespace=state["workspace_id"]
        )
        
        # 🚀 FIX: Added thresholding to Reports to reject random data
        valid_chunks = [m["metadata"]["text"] for m in search_results["matches"] if m.get("score", 0) >= 0.50 and "text" in m["metadata"]]
        section_context = "\n".join(valid_chunks)
        
        if not section_context.strip():
            final_combined_report += f"## {section}\n*Insufficient institutional data found for this specific metric in the uploaded files.*\n\n"
            continue

        # 🚀 FIX: Brutal anti-hallucination prompt for report sections
        prompt = f"""Write a formal NAAC accreditation report section for: '{section}'.
        
        CONTEXT FROM UPLOADED DOCUMENTS:
        {section_context}
        
        STRICT ANTI-HALLUCINATION RULES:
        1. You are a strict auditor. If the CONTEXT does NOT explicitly contain information relevant to the exact topic '{section}', DO NOT write a fake report.
        2. If data is irrelevant or missing, reply EXACTLY with: "Insufficient institutional data found for {section}."
        3. DO NOT force fit data (e.g., do not talk about Industrial Visits if the section is about the Library).
        4. Ground every single claim strictly in the context. Do not make generic positive statements unless proven by the context.
        5. Use a highly formal, academic tone with Markdown headers and bullet points.
        """
        response = llm.invoke(prompt)
        final_combined_report += response.content + "\n\n---\n\n"

    state["final_report"] = final_combined_report
    return state

# COMPILING THE REPORT GRAPH
report_workflow = StateGraph(ReportGraphState)
report_workflow.add_node("compiler", report_compiler_loop)
report_workflow.set_entry_point("compiler")
report_workflow.add_edge("compiler", END)
report_app = report_workflow.compile()


# =====================================================================
# ✨ PART 3: UTILITY FUNCTIONS (Refiners)
# =====================================================================

def refine_report_logic(current_content: str, instruction: str) -> str:
    """Manual refiner for Chat-based report editing."""
    print(f"✨ REFINER: Instruction -> {instruction}")
    system_prompt = f"""You are a senior NAAC Compliance Editor. 
    Modify the existing report text based on this instruction: '{instruction}'. 
    
    STRICT RULES:
    1. Preserve the Markdown structure.
    2. Do NOT hallucinate data not present in the original report.
    3. Improve clarity and professional vocabulary.
    
    ORIGINAL REPORT:
    {current_content}"""
    
    response = llm.invoke(system_prompt)
    return response.content