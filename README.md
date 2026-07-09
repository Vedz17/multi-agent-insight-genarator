# 🧠 InsightGen AI: Multi-Agent RAG Engine

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

This repository houses the core AI Engine for InsightGen. It is a highly optimized, deterministic Retrieval-Augmented Generation (RAG) pipeline orchestrated using LangGraph. The engine is explicitly designed to eliminate AI hallucinations and ensure 100% data-grounded institutional reporting.

## ✨ Core AI Architecture
- **Directed Acyclic Graph (DAG) Workflow:** Orchestrated via LangGraph with a 3-tier multi-agent system:
  1. **Researcher Agent:** Extracts highly relevant chunks from the vector database.
  2. **Writer Agent:** Synthesizes context into formal NAAC compliance formats.
  3. **Auditor Agent:** Validates the generated output against the strict source context, triggering a rewrite if hallucinations are detected.
- **Strict Namespace Isolation:** Pinecone vector embeddings are strictly segregated by `workspaceId` to guarantee zero data leakage between different institutional tenants.
- **Gatekeeper Anti-Noise Filter:** A pre-retrieval LLM loop that evaluates context validity. It enforces a strict similarity threshold (>= 0.40) and explicitly drops irrelevant data (e.g., song lyrics, recipes) before it reaches the generation pipeline.
- **Asymmetric RAG:** High-precision retrieval tuned for chatbot interactions and high-recall tuned for comprehensive report generation.

## 🛠️ Tech Stack
- **Framework:** FastAPI / Python
- **Orchestration:** LangChain & LangGraph
- **LLM Engine:** Llama-3.1-8b-instant (via Groq for ultra-low latency)
- **Embeddings:** Google Generative AI Embeddings (`models/gemini-embedding-001`)
- **Vector Database:** Pinecone

## 🚀 Step-by-Step Setup

**1. Clone the repository**
\`\`\`bash
git clone https://github.com/your-username/insightgen-ai-engine.git
cd insightgen-ai-engine
\`\`\`

**2. Create a Virtual Environment**
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
\`\`\`

**3. Install Dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**4. Set up Environment Variables**
Create a \`.env\` file in the root directory:
\`\`\`env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_ai_studio_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=naac-report-index
\`\`\`

**5. Run the AI Server**
\`\`\`bash
uvicorn main:app --reload --port 8000
\`\`\`
The AI Engine will now be streaming data at \`http://localhost:8000\`.

---
*Architected with 🩵 by [Vedant Bhamare] - AI Enthusiast & Full Stack Developer*
