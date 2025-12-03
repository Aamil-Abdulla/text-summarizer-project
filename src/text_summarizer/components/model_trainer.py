import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = "cpu"  # force CPU

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)

        dataset = load_from_disk(self.config.data_path)
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=2,              # shorter for CPU
            per_device_train_batch_size=4,   # small batch for CPU
            per_device_eval_batch_size=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir=os.path.join(self.config.root_dir, "logs"),
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=False,                      # CPU can't do fp16
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        model.save_pretrained(os.path.join(self.config.root_dir, 't5-summarizer'))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, 'tokenizer'))
        print(f"Model and tokenizer saved to {self.config.root_dir}")