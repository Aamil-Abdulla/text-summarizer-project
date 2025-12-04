# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Import your pipelines
from text_summarizer.pipeline.prediction import PredictionPipeline
from text_summarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

# ---------------------- FastAPI App ----------------------
app = FastAPI(
    title="Text Summarization API",
    description="API for training and predicting text summarization using a pre-trained model",
    version="1.0"
)

# ---------------------- Redirect root to docs ----------------------
@app.get("/", tags=["Authentication"])
async def index() -> RedirectResponse:
    """
    Redirect root URL to API docs
    """
    return RedirectResponse(url="/docs")


# ---------------------- Training Endpoint ----------------------
@app.get("/train", tags=["Training"])
async def train_model() -> JSONResponse:
    """
    Trigger model training pipeline
    """
    try:
        trainer = ModelTrainerTrainingPipeline()
        trainer.main()
        return JSONResponse(content={"message": "Training completed successfully!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# ---------------------- Prediction Endpoint ----------------------
class TextRequest(BaseModel):
    text: str


@app.post("/predict", tags=["Prediction"])
async def predict_text(request: TextRequest) -> JSONResponse:
    """
    Generate a summary for the provided text using the trained model
    """
    try:
        pipeline = PredictionPipeline()
        summary = pipeline.predict(request.text)
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------- Run the server ----------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
