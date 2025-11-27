import os
from  transformers import AutoTokenizer
from text_summarizer.logging import logger
from datasets import load_dataset, load_from_disk
from text_summarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        inputs = self.tokenizer(
            example_batch['dialogue'],
            max_length=512,
            padding='max_length',
            truncation=True
        )
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'],
                max_length=150,
                padding='max_length',
                truncation=True
            )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        logger.info(f"Loaded dataset from {self.config.data_path}")
        dataset_samsum_pt= dataset_samsum.map(
            self.convert_examples_to_features,
            batched=True
        )
    
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))

        