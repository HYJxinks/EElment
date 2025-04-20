# -*- coding: utf-8 -*-
# file: data_embedding.py
# time: 2024/9/25 0951
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import numpy as np
import pickle
from transformers import BertTokenizer
from pathlib import Path
from utils.config import DATASET_FILES

class ClassicWordEmbeddings(object):
    """
    Context free Word Embeddings
    Args:
        dataset_name: The dataset to be processed. The type is a list, such as ['14lap', 'train']。
    """
    def __init__(self,dataset_name:list,dataset,vocab, model_fname="./glove.42B.300d.txt",max_seq_len=128,embedding_dim=300):
        self.dataset_name =dataset_name
        self.dataset = dataset
        self.vocab = vocab        
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.model_fname = model_fname
    
    def set_model(self,model_fname):
        self.model_fname = model_fname
        
    def _load_word_vec(self):
        embedding_path = Path(self.model_fname)  
        if(not embedding_path.exists()):
            print(f'The specified word embedding file: {self.model_fname} was not found. \
                  Please use the set_model method to set the correct word embedding file path, similar to set_model("./glove.42B.300d.txt")')
            return None
        else:
            fin = open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
            word_vec = {}
            for line in fin:
                tokens = line.rstrip().split()
                word, vec = ' '.join(tokens[:-self.embed_dim]), tokens[-self.embed_dim:]
                if word in self.vocab.word2idx.keys():
                    word_vec[word] = np.asarray(vec, dtype='float32')
            return word_vec
    
    def build_embedding_matrix(self):
        embedding_path = Path(DATASET_FILES[self.dataset_name[0]][self.dataset_name[1]] + '.embedding.matrix')
        if(embedding_path.exists()):
            print('loading embedding_matrix:', embedding_path)
            embedding_matrix = pickle.load(open(embedding_path, 'rb'))
            return embedding_matrix
        else:
            print('loading word vectors...')
            embedding_matrix = np.zeros((len(self.vocab.word2idx) + 2, self.embed_dim))  # idx 0 and 1 are all-zeros
            word_vec = self._load_word_vec()
            if(word_vec is None):
                return None
            print('building embedding_matrix:', embedding_path)           
            for word,i in self.vocab.word2idx.items():
                if word in word_vec.keys():
                    embedding_matrix[i] = word_vec[word]
                else:
                    print('word not in vocab:', word)
                    return None
            pickle.dump(embedding_matrix, open(embedding_path, 'wb'))
            return embedding_matrix