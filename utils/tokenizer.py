# -*- coding: utf-8 -*-
# file: data_embedding.py
# time: 2024/9/25 01630
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import os
import pickle
import numpy as np
from transformers import BertTokenizer

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    """
        In models such as Transformer, sentences in the corpus need to have the same input length. 
        This function is used to truncate overly long sentences and pad overly short sentences, 
        in order to standardize the length of all sentences in the corpus to a specified value 'maxlen'
    Args:
        sequence(list): The sentence sequences to be processed which is consisted of sequences of integers that have already undergone word-to-index conversion.
        maxlen(int): Standard length of the sequence
        dtype(dtype): Specify the data type of elements in the output array 
        padding(str): Type of padding. It can take two values: 'post' and 'pre', with 'post' being the default. 
                      'post' indicates that the sequence is padded at the end with the 'value' to reach the specified 'maxlen' length. 
                      'pre' indicates that the padding should occur at the beginning of the sequence.
        truncating(str): Type of truncation. It can take two values: 'post' and 'pre', with 'post' being the default. 
                      'post' indicates that the head of the sequence is preserved up to the 'maxlen' length, 
                      and the excess tail is truncated and discarded. 
                      'pre' indicates that the excess length at the beginning of the sequence is truncated and discarded.
        value: Padding value with 0 being the default
    Returns:
        numpy array: The standard-length sequence formed after padding or truncation.
    """
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def build_tokenizer(fnames, vocab, max_seq_len, dat_fname):
    """
    Create or load a tokenizer.

    If the file specified by `dat_fname` exists, it loads the tokenizer from that file. Otherwise, it creates a new tokenizer based on the text data from the files listed in `fnames` and saves the tokenizer to the file specified by `dat_fname`.

    Parameters:
    - fnames: List of filenames containing text data used to train the tokenizer.
    - vocab: Vocabulary used to create the tokenizer.
    - max_seq_len: Maximum sequence length, the tokenizer will encode the text according to this parameter.
    - dat_fname: Filename for the tokenizer data. If the file exists, the tokenizer is loaded from this file; otherwise, the newly created tokenizer is saved to this file.

    Returns:
    - tokenizer: The created or loaded tokenizer.
    """
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len,vocab)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer

def _load_word_vec(path, word2idx=None, embed_dim=300):
    """
        Load word embedding vectors
    Args:
        path (str): word embedding vectors filename.
        word2idx(Dictionary): Mapping from words to unique IDs in the vocabulary
        embed_dim(int): Dimension of embedding vectors
    Returns:
        Tokenizer(Dictionary): Dictionary composed of words and their corresponding embedding vectors. 
    """
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

def build_embedding_matrix(word2idx,embed_dim, dat_fname, embeding_fname='./glove.42B.300d.txt'):
    """
        Create an embedding matrix for words in corpus
    Args:
        word2idx(Dictionary): Mapping from words to unique IDs in the vocabulary
        embeding_fname(str): A file containing static embedding vectors for words, such as glove.42B.300d.txt
        embed_dim(int): Dimension of embedding vectors
        dat_fname(str): Output filename for the serializing word embedding vectors
    Returns:
        embedding_matrix: A matrix containing embedding vectors for all words in the corpus. 
    """
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        word_vec = _load_word_vec(embeding_fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

class Tokenizer(object):
    """A storage class for an individual token  — i.e. a word, punctuation symbol, whitespace, etc.

    Args:
        max_seq_len(int): Maximum sequence length of a sentence processed by the tokenizer
        lower(Boolean): Whether all words should be converted to lowercase, with the default value being True.
    """    
    def __init__(self, max_seq_len, vocab, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.vocab = vocab
    
    def text_to_idx(self,text,reverse=False,padding='post',truncating='post'):
        """Convert a comment statement into a sequence of integer indices corresponding to the vocabulary.
            Args:
                text (str): a comment sentence
                reverse (bool, optional): Whether to generate in reverse order. Defaults to False.
                padding (str, optional): Type of padding.  Defaults to 'post'.
                truncating (str, optional): Type of truncating.  Defaults to 'post'.

            Returns:
                numpy array: The standard-length sequence formed after padding or truncation.
        """
        sequence = self.text_to_sequence(text)
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]

        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating,value=self.vocab.word2idx['<pad>'])
    
    def text_to_sequence(self,text):
        
        if self.lower:
            text = text.lower()
        words = text.split()
        sequence = [self.vocab.word2idx[w]  for w in words]
        return sequence
    
    def sequence_to_text(self, sequence):
        """
        将词表下标序列转换为对应的词组成的句子。

        参数:
        sequence (list): 包含词表下标的列表。

        返回:
        str: 由词表下标对应的词组成的句子。
        """
        words = [self.vocab.idx2word[idx] for idx in sequence]
        if len(sequence)>1:
            return " ".join(words)
        else:
            return words[0]
        

class Tokenizer4Bert:
    """The Token class used in the BERT model 
    
    Args:
        max_seq_len(int): Maximum sequence length of a sentence processed by the tokenizer
        pretrained_bert_name(str): The name of the pre-trained BERT model used in a model.
    """
    def __init__(self, max_seq_len, pretrained_bert_name):      #pretrained_bert_name预训练模型
        self.tokenizer = BertTokenizer.from_pretrained(
        "/home/zsc/.cache/huggingface/hub/models--bert-base-uncased/snapshots/new",
        local_files_only=True,
        token=False,        #将use_auth_token换成token
        revision="86b5e0934494bd15c9632b12f734a8a67f723594",
        #clean_up_tokenization = True,      #显式设置,使用了会报错
        )
        self.max_seq_len = max_seq_len


    def text_to_idx(self, text, reverse=False, padding='post', truncating='post'):
        """Convert a comment statement into a sequence of integer indices corresponding to the vocabulary.
        Args:
            text (str): a comment sentence
            reverse (bool, optional): Whether to generate in reverse order. Defaults to False.
            padding (str, optional): Type of padding.  Defaults to 'post'.
            truncating (str, optional): Type of truncating.  Defaults to 'post'.

        Returns:
            numpy array: The standard-length sequence formed after padding or truncation.
        """
        sequence = self.text_to_sequence(text)
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
    
    def text_to_sequence(self, text):
        """
        Convert a comment statement into a sequence of integer indices corresponding to the vocabulary.
        Args:
            text (str): a comment sentence
        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    
    def sequence_to_text(self, sequence):
        """
        Convert a sequence of integer indices back to the original text.
        Args:
            sequence (list): a list of integer indices corresponding to the vocabulary
        """
        tokens = self.tokenizer.convert_ids_to_tokens(sequence)
        return self.tokenizer.convert_tokens_to_string(tokens)
