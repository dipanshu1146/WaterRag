
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.upload import router as upload_router
from src.api.routes.ask import router as ask_router

app = FastAPI(
    title="Wastewater RAG API",
    description="Search wastewater documents and answer questions using RAG",
    version="0.2.0",
)

configured_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(dict.fromkeys([
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *configured_origins,
    ])),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(ask_router)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "application": "Wastewater RAG",
        "message": "Backend is running successfully.",
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Wastewater RAG",
        "message": "Backend is running successfully.",
    }


@app.get("/api")
def api_info():
    return {
        "name": "Wastewater RAG API",
        "version": "0.2.0",
    }
