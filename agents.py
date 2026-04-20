import os
import time
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END

load_dotenv()

# --- INITIALIZE LLM (GROQ) ---
# --- INITIALIZE LLM (GROQ) ---
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY"),
    max_retries=1
)

class GraphState(TypedDict):
    question: str
    context: str
    draft: str
    feedback: str
    iteration: int
    domain: str
    chat_history: list 
    workspace_id: str

# --- INITIALIZE PINECONE & EMBEDDINGS ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("naac-report-index")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def researcher_agent(state: GraphState) -> GraphState:
    print("---🔍 RESEARCHER AGENT : Searching Pinecone for context---")  
    question = state["question"]
    query_vector = embeddings.embed_query(question)
    search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)  

    search_results = index.query(
        vector=query_vector, 
        top_k=3, 
        include_metadata=True,
        namespace=state.get("workspace_id") 
    )

    extracted_texts = []
    for match in search_results["matches"]:
        if "text" in match["metadata"]:
            extracted_texts.append(match["metadata"]["text"])

    final_context = "\n\n---\n\n".join(extracted_texts)
    state["context"] = final_context
    return state    

def writer_agent(state: GraphState) -> GraphState:
    print(f"---✍️ WRITER AGENT (Iteration {state.get('iteration', 0) + 1}): Drafting Report---")
    question = state["question"]
    context = state["context"]
    
    # 🧠 HISTORY FORMATTING: Puraane messages ko ek string mein badlo
    chat_history_list = state.get("chat_history", [])
    history_text = ""
    if chat_history_list:
        for msg in chat_history_list:
            role = "USER" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"
    else:
        history_text = "No previous conversation."
    
    # 🛡️ THE IRON GUARDRAIL (Prompt Update)
    system_prompt = f"""You are an expert {state['domain']} Report Writer.
    Your job is to answer the user's query based ONLY on the provided Context and Chat History.
    
    STRICT RULES:
    1. THE IRON GUARDRAIL: Do not use outside knowledge. If the answer is not in the Context or Chat History, DO NOT guess. Simply reply: "I am sorry, but I can only answer questions based on the uploaded document or our current conversation."
    2. Format using bullet points or clean paragraphs.
    3. Act strictly as a professional {state['domain']} specialist.
    4. Use clean Markdown formatting (# for Titles, **Bold** for terms).
    
    PREVIOUS CHAT HISTORY:
    {history_text}
    
    CONTEXT FROM UPLOADED DOCUMENTS:
    {context}
    
    CURRENT USER QUERY:
    {question}
    """
    
    
    response = llm.invoke(system_prompt)
    
    state["draft"] = response.content
    state["iteration"] = state.get("iteration", 0) + 1 
    
    return state

def reviewer_agent(state: GraphState) -> GraphState:
    print("---🕵️ REVIEWER AGENT: Checking Quality---")
    question = state["question"]
    draft = state["draft"]
    
    system_prompt = f"""You are a senior {state['domain']} Reviewer.
    
    USER QUESTION: {question}
    DRAFTED REPORT: {draft}
    
    CHECKLIST:
    1. Does it answer the question? (Note: If Writer says 'data not available', this IS a valid answer. Approve it.)
    2. Is it professional?
    
    If it's good, reply with exactly: "PASS". 
    Otherwise, give specific feedback on what to fix.
    """
    
   
    response = llm.invoke(system_prompt)
    state["feedback"] = response.content.strip()
    return state

# 🚀 SMART ROUTER 
def review_router(state: GraphState):
    feedback = state.get("feedback", "").strip().upper()
    current_iteration = state.get("iteration", 0)
    
    is_approved = "PASS" in feedback
    print(f"Router Check -> Iteration: {current_iteration} | Approved: {is_approved}")

    if is_approved:
        print("✅ Reviewer Passed the Draft!")
        return "end_process"
    elif current_iteration >= 3:
        print("---⚠️ MAX ITERATIONS REACHED: Forcing Exit---")
        return "end_process"
    else:
        print(f"🔄 Reviewer rejected. Feedback: {feedback[:50]}...")
        return "rewrite_draft"
    
# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(GraphState)

workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")    
workflow.add_edge("writer", "reviewer")

workflow.add_conditional_edges(
    "reviewer", 
    review_router, 
    {"end_process": END, "rewrite_draft": "writer"}
)

app = workflow.compile()