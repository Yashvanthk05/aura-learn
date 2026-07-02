from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.routes import router, init_services

import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing services...")
    init_services()
    print("Services initialized successfully")
    yield
    print("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.DESCRIPTION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["API"])

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "status": "running",
        "docs_url": "/docs",
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )