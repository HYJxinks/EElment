# -*- coding: utf-8 -*-
# file: data_embedding.py
# time: 2024/9/25 01450
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import pickle
from pathlib import Path
from collections import Counter
from utils.config import DATASET_FILES


class Vocab(object):
    """
    Vocabulary class for mapping between words and indices.
    Args:
        dataset(str): The dataset to be processed, such as '14lap', '14lap'
        lower(Boolean): Whether all words should be converted to lowercase, with the default value being True.
        specials(List[str]): A list of special tokens that should be added to the vocabulary.
        wordidx(dict): A dictionary that maps words to indexes.
        idx2word(dict): A dictionary that maps indexes to words.
    
    """
    def __init__(self, dataset, lower=True, specials=["<pad>", "<unk>"]):
        self.dataset = dataset
        self.lower = lower
        self.specials = specials    
        self.word2idx = {}
        self.idx2word = specials
    
    def read_corpus(self):
        """
        Read corpus from a dataset.
        return:

            text(str): A text contains all sentences in corpus.
        """
        for fname in DATASET_FILES[self.dataset].values():
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            if 'inference' in fname:
                for i in range(0, len(lines)):
                    comment_text = lines[i].strip()
                    position = comment_text.find('$LABEL$') 
                    comment_text = comment_text[:position]
                    comment_text = comment_text.replace('[B-ASP]','')
                    comment_text = comment_text.replace('[E-ASP]','')
                    
                    comment_text = comment_text.strip()
                    if '-LRB-' in comment_text:
                        comment_text=comment_text.replace('-LRB-','(')
                    if '-RRB-' in comment_text:
                        comment_text=comment_text.replace('-RRB-',')')
            else: 
                for i in range(0, len(lines), 3):
                    text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
                    aspect = lines[i + 1].lower().strip()
                    text_raw = text_left + " " + aspect + " " + text_right
                    if '-LRB-' in text_raw:
                        text_raw=text_raw.replace('-LRB-','(')
                    if '-RRB-' in text_raw:
                        text_raw=text_raw.replace('-RRB-',')') 
                    comment_text += text_raw + " "
        return comment_text
        
    def build_vocab(self, text):
        """
        Build vocabulary from sentences.
        Args:
            text(str): A text of contains all sentences in corpus.
        """
        if self.lower:
            text = text.lower()
        counter = Counter(text.split())

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0]) # sort by word
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)  # sort by frequency

        for word, _ in words_and_frequencies:
            self.idx2word.append(word)

        # word2idx is simply a reverse dict for idx2word
        # 0 corresponds to <pad>, and 1 corresponds to <unk>.
        self.word2idx = {tok: i+len(self.specials) for i, tok in enumerate(self.idx2word)}
        for i,special in enumerate(self.specials):
            self.word2idx[special] = i

    def __eq__(self, other):
        if self.word2idx != other.word2idx:
            return False
        if self.idx2word != other.idx2word:
            return False
        return True

    def __len__(self):
        return len(self.idx2word)

    def extend(self, v):
        words = v.idx2word
        for w in words:
            if w not in self.word2idx:
                self.idx2word.append(w)
                self.word2idx[w] = len(self.word2idx) - 1
        return self
    
    @classmethod
    def load_vocab(cls,dataset):
        """
        Use this class method to load or create a vocabulary.
        Args:
            dataset(str): The dataset to be processed, such as '14lap', '14lap'
            sentences: A list of sentences.
        """
        vocab_file = Path(DATASET_FILES[dataset])
        vocab_file = vocab_file.parent / dataset+'.vocab'
        if vocab_file.exists():
            print('Vocab file exists, loading vocab from:', vocab_file)
            with open(vocab_file, "rb") as f:
                data = pickle.load(f)
            return cls(data)
        else:
            print('building vocab:', vocab_file)
            vocab = Vocab(dataset)
            text = vocab.read_corpus()
            vocab.build_vocab(text)
            vocab.save_vocab(vocab_file)
            return vocab                        

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            print('Saving vocab to:', vocab_path)
            pickle.dump(self, f)