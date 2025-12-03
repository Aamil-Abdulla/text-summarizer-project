from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import load_from_disk
from text_summarizer.logging import logger

class DataTransformation:
    def __init__(self, config):
        self.config = config
        # Use t5-small or t5-mini for CPU
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def convert_examples_to_features(self, example_batch):
        dialogues = [str(x) for x in example_batch['dialogue']]
        summaries = [str(x) for x in example_batch['summary']]

        inputs = self.tokenizer(
            dialogues,
            max_length=256,  # shorter for CPU
            truncation=True,
            padding='max_length'
        )

        # For labels, HF v5+ use text_target
        labels = self.tokenizer(
            summaries,
            max_length=64,  # shorter for CPU
            truncation=True,
            padding='max_length',
            text_target=summaries
        )

        inputs["labels"] = labels["input_ids"]
        return inputs

    def convert(self):
        logger.info(f"Loading raw dataset from: {self.config.train_path.parent}")
        dataset = load_dataset(
            "csv",
            data_files={
                "train": str(self.config.train_path),
                "validation": str(self.config.validation_path),
                "test": str(self.config.test_path)
            }
        )

        logger.info("Tokenizing dataset...")
        # Use num_proc to speed up tokenization
        tokenized_dataset = dataset.map(
            self.convert_examples_to_features,
            batched=True,
            num_proc=4,  # parallel CPU cores
            remove_columns=["dialogue", "summary"]
        )

        logger.info("Saving tokenized dataset...")
        tokenized_dataset.save_to_disk(self.config.root_dir)
        logger.info(f"Tokenized dataset saved at {self.config.root_dir}")
