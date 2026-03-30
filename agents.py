import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY"))

class GraphState(TypedDict):
    question: str
    context: str
    draft: str
    feedback: str
    iteration: int

pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index =pc.Index("naac-report-index")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def researcher_agent(state:GraphState) -> GraphState:
    print("---RESEARCHER AGENT : Searching Pinecone for context---")  
    
#READ USERS QUESTION FROM STATE
    question = state["question"]
    
#CONVERT QUESTION TO VECTOR
    query_vector = embeddings.embed_query(question)

#SEARCH PINECONE USING VECTOR
    search_results = index.query(vector=query_vector, top_k=3, include_metadata=True)  

#EXTRACT RELEVANT TEXT FROM SEARCH RESULTS AND PUT THEM INTO LIST
    extracted_texts=[]
    for match in search_results["matches"]:
     if "text" in match["metadata"]:
        extracted_texts.append(match["metadata"]["text"])

#JOIN EXTRACTED CHUNKS TO CREATE PARAGRAPH
    final_context = "\n\n---\n\n".join(extracted_texts)

#UPDATE STATE WITH CONTEXT
    state["context"] = final_context
    return state    

#WRITER AGENT: DRAFTING THE NAAC REPORT BASED ON THE CONTEXT AND QUESTION
def writer_agent(state: GraphState) -> GraphState:
    print("---✍️ WRITER AGENT: Drafting NAAC Report---")
    
    # 1. Read the Question and Context from the state
    question = state["question"]
    context = state["context"]
    
    # 2. PROMPT ENGINEERING (The NAAC Rules)
    system_prompt = f"""You are an expert NAAC Accreditation Report Writer for a prestigious college.
    Your job is to answer the user's query based ONLY on the provided context.
    
    STRICT RULES:
    1. Do not hallucinate or make up fake numbers. 
    2. If the answer is not in the context, clearly state: "The requested data is not available in the uploaded documents."
    3. Format the answer professionally using bullet points or clean paragraphs.
    
    CONTEXT FETCHED FROM DATABASE:
    {context}
    
    USER QUERY:
    {question}
    
    Now, write the draft response:"""
    
    # 3. Call the Gemini Model to write the draft
    response = llm.invoke(system_prompt)
    
    # 4. Save the generated text into the Notebook's 'draft' section
    state["draft"] = response.content
    
    return state

 #REVIEWER AGENT: CHECKING THE DRAFT AGAINST NAAC RULES
def reviewer_agent(state: GraphState) -> GraphState:
    print("---🕵️ REVIEWER AGENT: Checking Against NAAC Rules---")
    
    #  Read what the user asked and what the Writer drafted
    question = state["question"]
    draft = state["draft"]
    
    #  PROMPT ENGINEERING 
    system_prompt = f"""You are a strict, senior NAAC Compliance Reviewer.
    Your job is to evaluate the drafted report.
    
    USER QUESTION: {question}
    
    DRAFTED REPORT:
    {draft}
    
    CHECKLIST:
    1. Does the draft directly answer the user's question?
    2. Is it formatted cleanly and professionally (e.g., bullet points)?
    3. Is there any unnecessary conversational text like "Here is your answer"?
    
    If the draft is PERFECT and meets all criteria, reply with exactly one word: "PASS".
    If there are formatting issues or missing info, write strict feedback on what to fix. Do NOT rewrite the draft yourself, just give feedback.
    """
    
    # Call the LLM to inspect the draft
    response = llm.invoke(system_prompt)
    
    # Save the inspector's notes in the 'feedback' section of the notebook
    state["feedback"] = response.content.strip()
    
    return state

#initialise new graph
workflow=StateGraph(GraphState)

#adding nodes to the graph
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

#defining router
def reviewe_router(state:GraphState):
    feedback=state["feedback"]
    if state["feedback"]=="PASS":
        return END
    else:
        return "writer"
    
#adding edges to the graph
workflow.set_entry_point("researcher")
workflow.add_edge("researcher","writer")    
workflow.add_edge("writer","reviewer")

#CONDITIONAL EDGE
workflow.add_conditional_edges("reviewer", review_router,{"end_process": END, "rewrite_draft": "writer"})

app=workflow.compile()

if __name__ == "__main__":
    print("\n🚀 Starting Autonomous LangGraph Loop...\n")
    
    test_state = {
        "question": "What are the skills of Vedant Bhamare?", 
        "context": "",
        "draft": "",
        "feedback": "",
        "iteration": 0
    }
    
    # no manual function call , direct 'app' invoke the app
    final_output = app.invoke(test_state)
    
    print("\n✅ --- FINAL APPROVED DRAFT --- ✅\n")
    print(final_output["draft"])
    print("\n========================================================\n")
