from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initializing FastAPI application
app = FastAPI(
    title="Entity Inference APIs",
    description="Entity Inference APIs using AI models",
    version="0.1",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)


# Initialize AI Model
sentiment_model = pipeline(
    task="sentiment-analysis", 
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


# Data Models
class SentimentRequest(BaseModel):
    text: str 

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


# Route for sentiment analysis
@app.post(
    path="/analyze_sentiment", 
    status_code=status.HTTP_200_OK,
    response_model=SentimentResponse,
    tags=["Sentiment Analysis"]
)
def analyze_sentiment(request: SentimentRequest):   
    text = request.text
    try:
        result = sentiment_model(text)[0]   
        sentiment = result['label']
        confidence = result['score']
        response_data = SentimentResponse(
            text=text, sentiment=sentiment, confidence=confidence
        )
        return response_data
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    

# Health check endpoint
@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "OK"}


# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)