import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        import os

        print("### PredictionPipeline initializing... ###")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        model_dir = os.path.join(project_root, "artifacts/model_trainer/t5-summarizer")
        tokenizer_dir = os.path.join(project_root, "artifacts/model_trainer/tokenizer")

        print("MODEL PATH:", model_dir)
        print("TOKENIZER PATH:", tokenizer_dir)
        print("MODEL EXISTS:", os.path.exists(model_dir))
        print("TOKENIZER EXISTS:", os.path.exists(tokenizer_dir))

        # Temporary stop BEFORE loading - avoids crash
        if not os.path.exists(model_dir) or not os.path.exists(tokenizer_dir):
            print("❌ MODEL OR TOKENIZER MISSING → CRASH AVOIDED")
            raise RuntimeError("Model files missing")

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

        self.pipeline = pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1
        )


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

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
