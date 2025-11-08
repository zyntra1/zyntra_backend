from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import engine, Base
from app.routers import auth_router, admins_router, users_router

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Zyntra Backend API with Admin and User Management",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(admins_router)
app.include_router(users_router)


@app.get("/")
async def root():
    """
    Root endpoint - API health check.
    """
    return {
        "message": "Welcome to Zyntra API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
