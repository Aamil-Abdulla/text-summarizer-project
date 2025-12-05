# app.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
import uvicorn
import os
import sys

# Add src folder to Python path (needed for Render)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from text_summarizer.pipeline.prediction import PredictionPipeline

app = FastAPI(
    title="Text Summarization API",
    description="API for predicting summaries using a trained model",
    version="1.0"
)

# ---------------------- Redirect root to docs ----------------------
@app.get("/", tags=["Base"])
async def index():
    return RedirectResponse(url="/docs")

# ---------------------- Prediction Endpoint ----------------------
class TextRequest(BaseModel):
    text: str

@app.post("/predict", tags=["Prediction"])
async def predict_text(request: TextRequest):
    try:
        pipeline = PredictionPipeline()
        summary = pipeline.predict(request.text)
        return {"summary": summary}
    except Exception as e:
        return Response(f"Error during prediction: {e}")

# ---------------------- Run the server ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
