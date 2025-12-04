from text_summarizer.entity import ModelEvaluationConfig
from text_summarizer.logging import logger
import os
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from datasets import Dataset
from evaluate import load
import pandas as pd
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model safely on Windows
        model_dir = str(self.config.model_path.as_posix())
        tokenizer_dir = str(self.config.tokenizer_path.as_posix())

        logger.info(f"Loading model from {model_dir}")
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)

        logger.info(f"Loading tokenizer from {tokenizer_dir}")
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_dir)

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i:i + batch_size]

    def calculate_metrics_on_test_ds(
        self, dataset, metric, batch_size=4
    ):
        self.model.eval()
        preds, refs = [], []

        for batch in tqdm(self.generate_batch_sized_chunks(list(range(len(dataset))), batch_size)):
            texts = [dataset[i][self.config.column_text] for i in batch]
            summaries = [dataset[i][self.config.column_summary] for i in batch]

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    num_beams=2,
                    early_stopping=True
                )

            decoded_preds = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            preds.extend(decoded_preds)
            refs.extend(summaries)

        return metric.compute(predictions=preds, references=refs)

    
    def run(self):
        self.evaluate()

    def evaluate(self):
        logger.info("Loading ROUGE metric")
        metric = load("rouge")

        # Load test CSV directly
        test_csv_path = self.config.data_path
        if not test_csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
        logger.info(f"Loading test dataset from {test_csv_path}")
        df = pd.read_csv(test_csv_path)
        test_dataset = Dataset.from_pandas(df)

        logger.info("Calculating metrics on test dataset")
        results = self.calculate_metrics_on_test_ds(test_dataset, metric)
         

# inside evaluate() before saving
        self.config.metric_file_name.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([results]).to_csv(self.config.metric_file_name, index=False)
        logger.info(f"Saving metrics to {self.config.metric_file_name}")
        pd.DataFrame([results]).to_csv(self.config.metric_file_name, index=False)
        logger.info(f"Metrics saved to {self.config.metric_file_name}")
        logger.info("Evaluation complete!")