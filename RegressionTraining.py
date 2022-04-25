import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertConfig, BertForSequenceClassification, AutoModelForSequenceClassification, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
from scipy.stats import spearmanr
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.utils import shuffle
import re

parser = argparse.ArgumentParser()

parser.add_argument(“-p”, "--pretrained_model", help="path to the pretrained transformer", default='proberta4/checkpoint-1160000')
parser.add_argument(“-t”, "--train", help="path to the train dataset", default='Datasets/Degree_tokenized_split_three_ways/sorted_train.csv')
parser.add_argument(“-v”, "--valid", help="provide the path to the test dataset", default='Datasets/Degree_tokenized_split_three_ways/sorted_test.csv')
parser.add_argument(“-o”, "--output_dir", help="path to the output_dir", required=True)
parser.add_argument(“-fp”, "--fp16", help="fp16 mixed precision", action="store_true")
parser.add_argument(“-e”, "--epochs", help="number of training epochs", default=200, type=int)
parser.add_argument(“-lr”, "--learning_rate", help="learning rate", default=5e-7, type=float)
parser.add_argument(“-b”, "--batch_size", help="training batch size", default=256, type=int)
parser.add_argument(“-tk”, "--tokenizer", help="path to the tokenizer", default="proberta512")
parser.add_argument(“-m”, "--max_length", help="max length of the model", default=512, type=int)


args = parser.parse_args()

class ProteinDegreeDataset(Dataset):

    def __init__(self, split="train", max_length=args.max_length):
        self.trainFilePath = args.train
        self.validFilePath = args.valid
        if split=="train":
            self.seqs, self.labels = self.load_dataset(self.trainFilePath)
        else:
            self.seqs, self.labels = self.load_dataset(self.validFilePath)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer)

        self.max_length = max_length

    def load_dataset(self,path):
        df = pd.read_csv(path,names=['Labels', 'Sequence','Degree','Tokenized Sequence'],skiprows=1)

        df['Degree'] = np.log(df['Degree'])
        df['Degree'] = (df['Degree'] - np.mean(df['Degree']) )/ np.std(df['Degree'])
    
        seq = list(df['Sequence'])
        label = list(df['Degree'].astype(float))

        return seq, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        seq = " ".join("".join(self.seqs[idx].split()))
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['Labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return sample

train_dataset = ProteinDegreeDataset(split="train", max_length=args.max_length)
val_dataset = ProteinDegreeDataset(split="valid", max_length=max_length)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    meanSquareError = mean_squared_error(labels, preds)
    residuals = []
    for i in range(len(labels)):
        residuals.append(labels[i] - preds[i])
    df = pd.DataFrame({'freq': residuals})
    residualStandardDeviation = np.std(residuals)
    spearmancoeffiecient = spearmanr(labels, preds)
    return {
        'meanSquareError' : meanSquareError,
        'residualStandardDeviation' : residualStandardDeviation,
        'spearmanr' : spearmancoeffiecient[0]
    }

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=1)
    model.dropout = nn.Dropout(0.1)
    return model

training_args = TrainingArguments(
    output_dir=os.path.join('../models', args.output_dir)       
    num_train_epochs=args.epochs,              
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=16,   
    warmup_steps=1000,              
    learning_rate=args.learning_rate,
    logging_dir=os.path.join('../ModelLogs', args.output_dir)   ,           
    logging_steps=200,           
    do_train=True,          
    do_eval=True,                   
    evaluation_strategy="epoch",    
    gradient_accumulation_steps=args.batch_size,
    fp16=args.fp16,                      
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model_init=model_init,                
    args=training_args,                 
    train_dataset=train_dataset,      
    eval_dataset=val_dataset,          
    compute_metrics = compute_metrics      
)
if os.path.isdir(os.path.join('../models', args.output_dir)):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()