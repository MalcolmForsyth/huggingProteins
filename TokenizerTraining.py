from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='choose the algorithm for the tokenizer: BPE, UNI, or WPC', default='BPE')
parser.add_argument('--dataset', help='path to the dataset for training', default='../Datasets/UniRef90_Data/shrunked_valid.txt')
parser.add_argument('--output_dir', help='path to the output directory')

args = parser.parse_args()

alg, path, output_dir = args.alg, args.dataset, args.output_dir
unk_token = "<UNK>"
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]

def prepare(alg):
    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token = unk_token))
        trainer = BpeTrainer(special_tokens = spl_tokens)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token= unk_token, special_tokens = spl_tokens)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token = unk_token))
        trainer = WordPieceTrainer(special_tokens = spl_tokens)
    return tokenizer, trainer

def load_dataset(path):
    txt_file = open(path, "r")
    dataset = txt_file.readlines()
    return dataset

def train(dataset, tokenizer, trainer, output_dir):
    tokenizer.train_from_iterator(dataset, trainer=trainer)
    tokenizer.save('../Tokenizers/' + output_dir)
    tokenizer = Tokenizer.from_file('../Tokenizers/' + output_dir)
    return tokenizer

tokenizer, trainer = prepare(alg)
dataset = load_dataset(path)
print('training')
tokenizer = train(dataset, tokenizer, trainer, output_dir)

output = tokenize(dataset[0])
print(output)