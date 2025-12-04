from text_summarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        # Load tokenizer and pipeline once to avoid reloading every prediction
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.summarization_pipeline = pipeline(
            "summarization", 
            model=self.config.model_path, 
            tokenizer=self.tokenizer
        )

    def predict(self, text):
        if isinstance(text, list):
            text = " ".join(text)  # join list into one string
        print("Generating Summary...")
        summary = self.summarization_pipeline(
            text, 
            max_length=150, 
            min_length=30, 
            do_sample=False
        )
        return summary[0]['summary_text']
