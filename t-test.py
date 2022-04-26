import pandas as pd
import numpy as np
import transformers
import torch
import torch.nn as nn
import argparse
from scipy import stats
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, RobertaTokenizerFast
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()

parser.add_argument('-m', 'model_name', required=True)
parser.add_argument('-t', 'tokenizer_name', required=True)
parser.add_argument('-mx', 'max_length', required=True)

args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_name, max_length=args.max_length, do_lower_case=False)

model.to(device)
model.eval()
model.zero_grad()

df = pd.read_csv('DegreesWithMultipleAugmentations.csv',index_col=0)
sample = df.sample(frac=0.001, replace=True)
rejected = 0
total = 0
for s in sample['Sequence']:
    inputs = tokenizer(s, return_tensors='pt', truncation=True, max_length=args.max_length)
    inputs.to(device)
    outputs = model(inputs['input_ids'], output_attentions=True)
    attention = outputs[1]
    summed = attention[j]
    num = summed.cpu().detach().numpy()
    last_attention = num[0][0]
    last_attention = np.log(last_attention)
    
    for data in last_attention:
        l_positions = [pos for pos, char in enumerate(s) if char == 'L']
        l_data = data[l_positions]
        t_value,p_value=stats.ttest_1samp(l_data,np.mean(data))
        if p_value/2 <= 0.05:
            rejected += 1
        total += 1
print("proportion of null hypothesis rejected: " + str(rejected / total))
            

    

