import os
from transformers import AutoTokenizer, pipeline

class PredictionPipeline:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "../../artifacts/model_trainer/t5-summarizer")
        tokenizer_dir = os.path.join(base_dir, "../../artifacts/model_trainer/tokenizer")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self.summarization_pipeline = pipeline(
            "summarization",
            model=model_dir,
            tokenizer=self.tokenizer,
            device=-1  # CPU
        )

    def predict(self, text):
        if isinstance(text, list):
            text = " ".join(text)
        summary = self.summarization_pipeline(
            text,
            max_length=150,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']
