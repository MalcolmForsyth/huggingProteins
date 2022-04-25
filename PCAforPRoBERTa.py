from pandas.core.frame import DataFrame
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import pipeline
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from transformers import BertTokenizer, BertForMaskedLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--pretrained_model", help="path to the pretrained transformer", default='../models/proberta512/checkpoint-1160000')
parser.add_argument('-tk', "--tokenizer", help="path to the tokenizer", default='../models/proberta512')
parser.add_argument('-m', "--max_length", help="max length of the model", default=512, type=int)

args = parser.parse_args()

max_length = 512
tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join("../models", args.tokenizer), max_len=args.max_length, truncation=True)
model = RobertaForMaskedLM.from_pretrained(os.path.join("../models", args.pretrained_model)

sample_size = 10000
dataset = pd.read_csv("../Datasets/Finetune_fam_data_500K.csv")
seq_list = list(dataset['Tokenized Sequence'])
for i in range(len(seq_list)):
    element = seq_list[i]
    #remove the space in join for our pretrained model, check the space error in the finetuning scripts
    #add the space in join for rostlab
    seq_list[i] = "".join(element[1:].split(" "))
    seq_list[i] = seq_list[i][:max_length - 1]
labels = list(dataset['Protein families'][:sample_size])
hidden_size = model.hidden_size
feat_cols = ['cls weight '+str(i) for i in range(hidden_size)]
df = pd.DataFrame(columns=feat_cols)
for i in range(sample_size):
    print(i)
    input = tokenizer(seq_list[i], return_tensors="pt").to('cuda')
    output = model(**input, output_hidden_states=True)
    df.loc[i] = output['hidden_states'][-1][0][0].detach().cpu().numpy()
df['labels'] = labels

pca = PCA(n_components=50)
pca_result = pca.fit_transform(df[feat_cols].values)

np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

#plots
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=pca_result[:,0], y=pca_result[:,1],
    hue=labels,
    legend="full",
    alpha=0.6
)

#save picture
plt.savefig('pca.png')

tsne = TSNE(n_components=2)

tsne_result = tsne.fit_transform(pca_result)
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=tsne_result[:,0], y = tsne_result[:,1],
    hue=labels,
    legend='full',
    alpha=0.6
)

plt.savefig('tsne.png')
df = DataFrame[data, columns = feat_cols]
pca = PCA(n_components=50)
pca_result = pca.fit_transform(df[feat_cols].values)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
'''