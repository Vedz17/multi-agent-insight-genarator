from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load secret keys from .env file
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="Multi-Agent Insight Generator",
    description="AI Engine for NAAC Compliance Reports",
    version="1.0.0"
)

# Configure CORS (Allows Next.js Frontend to talk to this Python Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev. In production, we will lock this to your frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Route
@app.get("/")
async def root():
    return {
        "status": "Online", 
        "message": "The AI Engine is Live! 🚀",
        "system": "Awaiting NAAC PDFs..."
    }