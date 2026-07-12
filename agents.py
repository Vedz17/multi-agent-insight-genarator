import os
import time
from typing import TypedDict
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END
import cohere

load_dotenv()

# =====================================================================
#  INITIALIZATIONS (Shared Resources)
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
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# --- COHERE RERANKER ---
cohere_client = cohere.Client(
    api_key=os.getenv("COHERE_API_KEY")
)

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
#  PART 1: CHATBOT AGENTS (Optimized for Fast Insights)
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
    """Fetches chunks from Pinecone and Re-ranks them using Cohere."""

    print(
        f"--- CHAT RESEARCHER: Searching "
        f"[Namespace: {state.get('workspace_id')}]---"
    )

    question = state["question"]
    query_vector = embeddings.embed_query(question)

    # 1. PULL A WIDE NET FROM PINECONE (Top 20)
    search_results = index.query(
        vector=query_vector,
        top_k=20,
        include_metadata=True,
        namespace=state.get("workspace_id")
    )

    raw_matches = search_results.get("matches", [])

    raw_chunks = [
        m["metadata"]["text"]
        for m in raw_matches
        if "text" in m["metadata"]
    ]

    if not raw_chunks:
        state["context"] = "NO_RELEVANT_DATA"
        return state

    print(" CHAT RESEARCHER: Reranking with Cohere...")

    try:
        # 2. Score and sort chunks based on relevance
        rerank_results = cohere_client.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=raw_chunks,
            top_n=5
        )

        # 3. Keep sufficiently relevant chunks
        valid_chunks = [
            raw_chunks[res.index]
            for res in rerank_results.results
            if res.relevance_score >= 0.05
        ]

    except Exception as e:
        print(f" Cohere API Error: {e}")
        valid_chunks = raw_chunks[:5]

    if not valid_chunks:
        print(
            " WARNING: Cohere rejected all chunks. "
            "Probably irrelevant data."
        )

        state["context"] = "NO_RELEVANT_DATA"
        return state

    state["context"] = "\n\n---\n\n".join(valid_chunks)

    return state


def writer_agent(state: GraphState) -> GraphState:
    """Synthesizes an analyst-style response."""

    print("--- CHAT WRITER: Drafting Response---")

    if state["context"] == "NO_RELEVANT_DATA":
        state["draft"] = (
            "System Block: Based on the uploaded documents, "
            "I could not find relevant institutional data to answer "
            "this query. Please ensure valid compliance files are uploaded."
        )

        state["iteration"] = state.get("iteration", 0) + 1
        return state

    question = state["question"]
    context = state["context"]

    chat_history_list = state.get("chat_history", [])

    history_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in chat_history_list
    ])

    system_prompt = f"""You are the InsightGen Intelligent Analyst.
Provide clear, structured insights based ONLY on the provided institutional documents.

STRICT ANTI-HALLUCINATION RULES:
1. STRICT GROUNDING: Use ONLY the context below. Do not use your pre-trained knowledge.
2. If the context has absolutely nothing to do with the user's query, reply EXACTLY with: "I cannot find specific data regarding this in the uploaded documents."
3. DO NOT force connections.
4. Do not guess the institution's name unless explicitly written in the context.
5. Use clean Markdown/bullets and maintain a professional tone.

DOCUMENTS CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUERY:
{question}
"""

    response = llm.invoke(system_prompt)

    state["draft"] = response.content
    state["iteration"] = state.get("iteration", 0) + 1

    return state


def reviewer_agent(state: GraphState) -> GraphState:
    """Quick check for hallucinations or missing info."""

    print("--- CHAT REVIEWER: Validating Accuracy---")

    system_prompt = f"""Review this response as a strict compliance auditor.

Does it accurately answer '{state['question']}' using ONLY the provided context?
Does it hallucinate any names, places, or metrics?

Reply 'PASS' if it is 100% accurate and grounded,
or provide critical feedback if it is hallucinating.

DRAFT:
{state['draft']}
"""

    response = llm.invoke(system_prompt)

    state["feedback"] = response.content.strip()

    return state


def review_router(state: GraphState):
    """Controls the loop - Limited to 1 pass for Chat speed."""

    if (
        "PASS" in state.get("feedback", "").upper()
        or state.get("iteration", 0) >= 1
    ):
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

workflow.add_conditional_edges(
    "reviewer",
    review_router,
    {
        "end_process": END,
        "rewrite_draft": "writer"
    }
)

app = workflow.compile()


# =====================================================================
#  PART 2: REPORT GENERATOR AGENTS (Section-by-Section)
# =====================================================================

class SectionGraphState(TypedDict):
    workspace_id: str
    criterion_id: int
    section_name: str
    context: str
    draft: str
    feedback: str
    review_status: str
    iteration: int


# ---------------------------------------------------------------------
# 🔍 REPORT RETRIEVER
# ---------------------------------------------------------------------

def report_retriever_agent(
    state: SectionGraphState
) -> SectionGraphState:
    """Retrieves and reranks evidence for one NAAC report section."""

    section_name = state["section_name"]
    workspace_id = state["workspace_id"]

    print(
        f"---🔍 REPORT RETRIEVER: Searching '{section_name}' "
        f"[Namespace: {workspace_id}]---"
    )

    # 1. Convert the section name into an embedding
    query_vector = embeddings.embed_query(section_name)

    # 2. Broad semantic retrieval from the correct workspace
    search_results = index.query(
        vector=query_vector,
        top_k=20,
        include_metadata=True,
        namespace=workspace_id
    )

    # 3. Extract text chunks safely
    raw_matches = search_results.get("matches", [])

    raw_chunks = [
        match["metadata"]["text"]
        for match in raw_matches
        if match.get("metadata")
        and "text" in match["metadata"]
    ]

    if not raw_chunks:
        print(
            f" REPORT RETRIEVER: "
            f"No evidence found for '{section_name}'"
        )

        state["context"] = "NO_RELEVANT_DATA"

        return state

    print(
        f" REPORT RETRIEVER: Reranking "
        f"{len(raw_chunks)} chunks with Cohere..."
    )

    try:
        # 4. Rerank against the exact NAAC section
        rerank_results = cohere_client.rerank(
            model="rerank-english-v3.0",
            query=section_name,
            documents=raw_chunks,
            top_n=5
        )

        # 5. Keep sufficiently relevant evidence
        valid_chunks = [
            raw_chunks[result.index]
            for result in rerank_results.results
            if result.relevance_score >= 0.05
        ]

    except Exception as e:
        print(
            f"🚨 REPORT RETRIEVER: "
            f"Cohere error: {e}"
        )

        print(
            "⚠️ REPORT RETRIEVER: "
            "Falling back to Pinecone top 5."
        )

        valid_chunks = raw_chunks[:5]

    if not valid_chunks:
        print(
            f"⚠️ REPORT RETRIEVER: "
            f"No relevant reranked evidence for '{section_name}'"
        )

        state["context"] = "NO_RELEVANT_DATA"

        return state

    # 6. Save evidence into LangGraph state
    state["context"] = "\n\n---\n\n".join(valid_chunks)

    print(
        f"✅ REPORT RETRIEVER: Selected "
        f"{len(valid_chunks)} evidence chunks."
    )

    return state


# ---------------------------------------------------------------------
#  REPORT WRITER
# ---------------------------------------------------------------------

def report_writer_agent(
    state: SectionGraphState
) -> SectionGraphState:
    """
    Writes or rewrites one grounded NAAC report section.

    The Writer may professionally paraphrase and synthesize retrieved
    institutional evidence, but must preserve its factual meaning.
    """

    section_name = state["section_name"]
    context = state["context"]
    iteration = state.get("iteration", 0)

    print(
        f"---✍️ REPORT WRITER: Drafting '{section_name}' "
        f"[Attempt: {iteration + 1}]---"
    )

    # No evidence means no LLM generation call
    if context == "NO_RELEVANT_DATA":
        state["draft"] = (
            f"## {section_name}\n"
            "*Insufficient institutional data found "
            "in the uploaded documents.*"
        )

        state["feedback"] = (
            "No relevant institutional evidence was retrieved."
        )

        state["review_status"] = "insufficient_data"

        return state

    # ================================================================
    # FIRST WRITER ATTEMPT
    # ================================================================

    if iteration == 0:
        prompt = f"""You are an expert NAAC Accreditation Report Writer.

Your task is to transform retrieved institutional evidence into a coherent,
professional NAAC report narrative for the section:

'{section_name}'

GROUNDING RULES:
1. Base the report ONLY on the institutional evidence provided below.
2. Preserve the factual meaning of the evidence.
3. You MAY professionally paraphrase the evidence.
4. You MAY combine closely related evidence into coherent paragraphs.
5. You MAY improve transitions, sentence flow, and professional NAAC-style language.
6. Supported names, dates, numbers, percentages, metrics, activities, and outcomes MAY be used when they are present in the evidence.
7. Do NOT invent unsupported institutional facts, quantitative claims, activities, achievements, or outcomes.
8. Do NOT turn a possible implication into a confirmed institutional fact.
9. Do NOT exaggerate the impact or benefit of an activity unless the evidence supports that impact.
10. Ignore clearly irrelevant or non-academic text.

WRITING RULES:
1. Start the response with exactly: ## {section_name}
2. Write a polished and human-readable institutional narrative.
3. Do not merely copy raw evidence line by line.
4. Avoid conversational phrases such as "Here is the report".
5. Do not mention the retrieved context, documents, or evidence in the final report.
6. Do not add a conclusion claiming excellence, success, effectiveness, or impact unless supported by the evidence.

INSTITUTIONAL EVIDENCE:
{context}

Write the grounded NAAC report section now."""

    # ================================================================
    # ONE CORRECTIVE REWRITE
    # ================================================================

    else:
        prompt = f"""You are an expert NAAC Accreditation Report Writer revising a draft after a factual grounding review.

Your task is to correct the report section:

'{section_name}'

IMPORTANT:
The previous draft was reviewed for MATERIAL factual grounding issues.
Correct the issues identified by the reviewer while preserving a polished,
professional NAAC narrative.

GROUNDING RULES:
1. Use ONLY the institutional evidence provided below.
2. Preserve the factual meaning of the evidence.
3. Professional paraphrasing is allowed.
4. Combining closely related evidence into coherent paragraphs is allowed.
5. Supported names, dates, numbers, percentages, metrics, activities, and outcomes MAY remain in the report.
6. Correct or remove factual claims specifically identified as unsupported or meaning-distorting by the reviewer.
7. Do NOT invent replacement facts while correcting the draft.
8. Do NOT convert implications or assumptions into confirmed institutional facts.
9. Do NOT exaggerate institutional outcomes or benefits beyond the evidence.
10. Keep valid and grounded parts of the previous draft wherever possible.

WRITING RULES:
1. Start the response with exactly: ## {section_name}
2. Return the COMPLETE corrected report section.
3. Maintain professional NAAC-style language and natural paragraph flow.
4. Do not explain what you changed.
5. Do not mention the reviewer, retrieved evidence, or correction process.

INSTITUTIONAL EVIDENCE:
{context}

PREVIOUS DRAFT:
{state.get("draft", "")}

GROUNDING REVIEW FEEDBACK:
{state.get("feedback", "")}

Write the complete corrected NAAC report section now."""

    response = llm.invoke(prompt)

    state["draft"] = response.content

    # iteration represents the number of Writer attempts
    state["iteration"] = iteration + 1

    state["review_status"] = "pending_review"

    return state

# ---------------------------------------------------------------------
#  REPORT REVIEWER
# ---------------------------------------------------------------------

def report_reviewer_agent(
    state: SectionGraphState
) -> SectionGraphState:
    """
    Reviews one NAAC report section only for material factual grounding issues.

    The Reviewer is not a grammar or style critic.
    Professional paraphrasing and synthesis are allowed as long as the
    factual meaning of the institutional evidence is preserved.
    """

    print(
        f"---🕵️ REPORT REVIEWER: Auditing "
        f"'{state['section_name']}'---"
    )

    # Retriever already confirmed that evidence is unavailable
    if state["context"] == "NO_RELEVANT_DATA":
        state["review_status"] = "insufficient_data"

        return state

    review_prompt = f"""You are a factual grounding auditor for a NAAC institutional report.

Your ONLY responsibility is to detect MATERIAL factual grounding problems.

SECTION:
{state['section_name']}

INSTITUTIONAL EVIDENCE:
{state['context']}

DRAFT:
{state['draft']}

IMPORTANT REVIEW PRINCIPLE:

The draft does NOT need to copy the institutional evidence word-for-word.

Professional paraphrasing, natural transitions, combining related facts,
and polished NAAC-style narrative are ALLOWED.

Judge SEMANTIC FAITHFULNESS, not lexical similarity.

FAIL THE DRAFT ONLY IF ONE OR MORE OF THESE PROBLEMS EXIST:

1. MATERIAL UNSUPPORTED FACT:
   The draft presents a specific institutional fact, activity, achievement,
   policy, outcome, or claim that is not supported by the evidence.

2. UNSUPPORTED QUANTITATIVE CLAIM:
   The draft introduces a number, percentage, metric, ranking, date, count,
   or measurable result that is not supported by the evidence.

3. MEANING DISTORTION:
   The draft materially changes the meaning of the evidence or turns an
   assumption, possibility, or implication into a confirmed fact.

4. MAJOR SECTION MISMATCH:
   A substantial part of the draft discusses information unrelated to the
   requested section or unsupported by the retrieved evidence.

DO NOT FAIL THE DRAFT FOR:

- Professional paraphrasing.
- Different wording from the evidence.
- Combining related evidence into one paragraph.
- Natural transitions between supported facts.
- Minor stylistic preferences.
- Grammar choices that do not alter factual meaning.
- General professional NAAC-style phrasing.
- Supported names, dates, numbers, percentages, or metrics.
- A reasonable summary of facts already present in the evidence.

REVIEW OUTPUT RULES:

If the draft is materially grounded and semantically faithful, reply EXACTLY:

PASS

If a material grounding problem exists, reply in this format:

FAIL

UNSUPPORTED CLAIM: "<copy the exact problematic words or sentence from the DRAFT>"
REASON: <briefly explain why the institutional evidence does not support it>

If there are multiple material problems, repeat the UNSUPPORTED CLAIM and REASON pair.

CRITICAL RULES:
1. The text inside UNSUPPORTED CLAIM must appear VERBATIM in the DRAFT.
2. Never claim that the draft contains a name, number, date, metric, activity,
   or statement unless that exact claim actually appears in the DRAFT.
3. Do not invent reviewer criticism.
4. Do not suggest stylistic rewrites.
5. Do not fail a draft merely because you would personally phrase it differently.
6. Be strict about factual grounding, but tolerant of professional writing freedom.

Audit the draft now."""

    response = llm.invoke(review_prompt)

    feedback = response.content.strip()

    state["feedback"] = feedback

    if feedback.upper() == "PASS":
        state["review_status"] = "approved"

    else:
        state["review_status"] = "needs_revision"

    return state

# ---------------------------------------------------------------------
#  REPORT FLAGGED NODE
# ---------------------------------------------------------------------

def report_flagged_agent(
    state: SectionGraphState
) -> SectionGraphState:
    """Marks a section as flagged after the maximum rewrite attempts."""

    print(
        f"---⚠️ REPORT FLAGGED: Maximum rewrite attempts reached "
        f"for '{state['section_name']}'---"
    )

    state["review_status"] = "flagged"

    return state


# ---------------------------------------------------------------------
# 🚦 REPORT REVIEW ROUTER
# ---------------------------------------------------------------------

def report_review_router(state: SectionGraphState):
    """Allows at most one corrective rewrite."""

    # No evidence was available
    if state.get("review_status") == "insufficient_data":
        return "insufficient_data"

    # Reviewer approved the section
    if state.get("review_status") == "approved":
        return "approved"

    # Writer has run fewer than 2 times
    if state.get("iteration", 0) < 2:
        return "rewrite"

    # Second Writer attempt also failed review
    return "flagged"


# ---------------------------------------------------------------------
# 🧠 COMPILE NEW SINGLE-SECTION LANGGRAPH
# ---------------------------------------------------------------------

section_workflow = StateGraph(SectionGraphState)

section_workflow.add_node(
    "retriever",
    report_retriever_agent
)

section_workflow.add_node(
    "writer",
    report_writer_agent
)

section_workflow.add_node(
    "reviewer",
    report_reviewer_agent
)

section_workflow.add_node(
    "flagged",
    report_flagged_agent
)

section_workflow.set_entry_point("retriever")

section_workflow.add_edge(
    "retriever",
    "writer"
)

section_workflow.add_edge(
    "writer",
    "reviewer"
)

section_workflow.add_conditional_edges(
    "reviewer",
    report_review_router,
    {
        "approved": END,
        "rewrite": "writer",
        "flagged": "flagged",
        "insufficient_data": END
    }
)

section_workflow.add_edge(
    "flagged",
    END
)

section_app = section_workflow.compile()


# =====================================================================
# 🛡️ LEGACY FULL-REPORT FLOW
# Kept temporarily so existing /generate-report still works.
# =====================================================================

class ReportGraphState(TypedDict):
    workspace_id: str
    criterion_id: int
    criterion_topics: str
    final_report: str


def report_compiler_loop(
    state: ReportGraphState
) -> ReportGraphState:

    print(
        f"---📝 SECTION COMPILER: "
        f"Criterion {state['criterion_id']}---"
    )

    full_topics = state["criterion_topics"].split(":")

    criterion_name = full_topics[0]

    sub_sections = [
        s.strip()
        for s in full_topics[1].split(",")
    ]

    final_combined_report = (
        f"# {criterion_name}\n\n"
    )

    for section in sub_sections:

        query_vector = embeddings.embed_query(section)

        # 1. Search
        search_results = index.query(
            vector=query_vector,
            top_k=8,
            include_metadata=True,
            namespace=state["workspace_id"]
        )

        # 2. Extract chunks
        raw_chunks = [
            match["metadata"]["text"]
            for match in search_results["matches"]
            if "text" in match["metadata"]
        ]

        # 3. Legacy LLM relevance gatekeeper
        context_check_prompt = f"""You are a data filter. Analyze this context and decide if it contains valid academic/institutional data for '{section}'.

CONTEXT:
{' '.join(raw_chunks[:3])}

REPLY ONLY 'YES' if it contains academic data,
or 'NO' if it is lyrics, recipes, or irrelevant noise.
"""

        is_relevant = (
            llm.invoke(context_check_prompt)
            .content
            .strip()
        )

        if "NO" in is_relevant.upper():

            final_combined_report += (
                f"## {section}\n"
                "*Insufficient institutional data found.*\n\n"
            )

            continue

        # 4. Legacy writer
        prompt = f"""Write a formal NAAC report section for: '{section}'.

Use ONLY this valid academic context:
{' '.join(raw_chunks)}

IF the context contains irrelevant lyrics or non-academic text,
IGNORE IT COMPLETELY.
"""

        response = llm.invoke(prompt)

        final_combined_report += (
            response.content
            + "\n\n---\n\n"
        )

    state["final_report"] = final_combined_report

    return state


# COMPILING LEGACY REPORT GRAPH
report_workflow = StateGraph(ReportGraphState)

report_workflow.add_node(
    "compiler",
    report_compiler_loop
)

report_workflow.set_entry_point("compiler")

report_workflow.add_edge(
    "compiler",
    END
)

report_app = report_workflow.compile()


# =====================================================================
# ✨ PART 3: UTILITY FUNCTIONS (Refiners)
# =====================================================================

def refine_report_logic(
    current_content: str,
    instruction: str
) -> str:
    """Manual refiner for Chat-based report editing."""

    print(
        f"✨ REFINER: Instruction -> {instruction}"
    )

    system_prompt = f"""You are a senior NAAC Compliance Editor.

Modify the existing report text based on this instruction:
'{instruction}'.

STRICT RULES:
1. Preserve the Markdown structure.
2. Do NOT hallucinate data not present in the original report.
3. Improve clarity and professional vocabulary.

ORIGINAL REPORT:
{current_content}"""

    response = llm.invoke(system_prompt)

    return response.content