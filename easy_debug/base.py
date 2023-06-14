import transformers
from datasets import load_dataset
import torch


def train(model_path, output_dir, train_data):
    dtype = torch.bfloat16
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    model = model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True, device_map="auto",
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            gradient_accumulation_steps=10,
            num_train_epochs=1,
            learning_rate=0.01,
            output_dir=output_dir
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )
    trainer.train()
