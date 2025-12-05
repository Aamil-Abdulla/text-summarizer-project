from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
import uvicorn
import os

from text_summarizer.pipeline.prediction import PredictionPipeline
from text_summarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

app = FastAPI(
    title="Text Summarization API",
    description="API for training and predicting text summarization using a pre-trained model",
    version="1.0"
)

# ---------------------- Redirect root to docs ----------------------
@app.get("/", tags=["Authentication"])
async def index():
    return RedirectResponse(url="/docs")

# ---------------------- Training Endpoint ----------------------
@app.get("/train", tags=["Training"])
async def train_model():
    try:
        trainer = ModelTrainerTrainingPipeline()
        trainer.main()
        return Response("Training Successful!!")
    except Exception as e:
        return Response(f"Error Occurred during training: {e}")

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
        return Response(f"Error Occurred during prediction: {e}")

# ---------------------- Run the server ----------------------
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # fallback to 8080 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
