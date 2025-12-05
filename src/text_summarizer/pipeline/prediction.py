import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        # project root = directory containing 'src'
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

        model_dir = os.path.join(project_root, "artifacts/model_trainer/t5-summarizer")
        tokenizer_dir = os.path.join(project_root, "artifacts/model_trainer/tokenizer")

        # Load tokenizer + model from local folders
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

    def predict(self, text):
        if isinstance(text, list):
            text = " ".join(text)

        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=30,
            do_sample=False
        )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
