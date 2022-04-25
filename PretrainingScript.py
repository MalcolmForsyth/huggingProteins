from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import datasets
import torch
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(“-t”, "--train", help="path to the train dataset", default='/tokenized_train_dataset_proberta2')
parser.add_argument(“-v”, "--valid", help="provide the path to the test dataset", default='tokenized_validation_dataset_proberta2')
parser.add_argument(“-o”, "--output_dir", help="path to the output_dir", required=True)
parser.add_argument(“-fp”, "--fp16", help="fp16 mixed precision", action="store_true")
parser.add_argument(“-e”, "--epochs", help="number of training epochs", default=200, type=int)
parser.add_argument(“-lr”, "--learning_rate", help="learning rate", default=1.5e-4, type=float)
parser.add_argument(“-b”, "--batch_size", help="training batch size", default=256, type=int)
parser.add_argument(“-tk”, "--tokenizer", help="path to the tokenizer", default="proberta512")
parser.add_argument(“-m”, "--max_length", help="max length of the model", default=512, type=int)
parser.add_argument(“-h”, "--hidden_layers", help="number of hidden layers", default=6, type=int)
parser.add_argument(“-ah”, "--attention_heads", help="number of attention heads", default=12, type=int)


args = parser.parse_args()

max_length = args.max_length
tokenized_train_dataset = load_from_disk(os.path.join('../Datasets', args.train))
tokenized_validation_dataset = load_from_disk(os.path.join('../Datasets', args.valid))
tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])
tokenized_validation_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'text'])

config = RobertaConfig(
    vocab_size=10_000,
    max_position_embeddings=max_length + 2,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join("../models", args.tokenizer), max_len=max_length)
model = RobertaForMaskedLM(config=config)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=os.path.join('../models', args.output_dir),
    overwrite_output_dir=False,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    eval_steps=10_000,
    evaluation_strategy="steps",
    metric_for_best_model="loss",
    save_steps=500,
    save_total_limit=100,
    prediction_loss_only=True,
    load_best_model_at_end=True,
    fp16=args.fp16,
    learning_rate=args.lr,
    logging_dir=os.path.join('../ModelLogs', args.output_dir)
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_train_dataset['train'],
    eval_dataset=tokenized_validation_dataset['train']
)

if os.path.isdir(os.path.join('../models', args.output_dir)):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()