# Importing FastAPI packages
from fastapi import FastAPI


app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc",
    title="Entity Inference APIs",
    description="Entity Inference APIs using AI models",
    version="0.1a",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)


# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Entity Inference APIs application is up and running!"}