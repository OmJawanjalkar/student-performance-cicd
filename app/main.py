from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="Student Performance Prediction API",
    description="Machine Learning API built with FastAPI",
    version="1.0.0",
)

app.include_router(router)