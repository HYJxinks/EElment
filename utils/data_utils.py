# -*- coding: utf-8 -*-
# file: data_utils.py
# time: 2024/7/24 0901
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import numpy as np
from torch.utils.data import Dataset
from utils.tokenizer import pad_and_truncate
from utils.config import DICT_POLARITY


class ABSADataset(Dataset):
    """The class used to read and store training, validation, or test data from a file

    Args:
        Dataset (Dataset): superclass
    """
    def __init__(self, fname,tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            
            comment_text = text_left +" "+ aspect +" "+ text_right
            text_indices = tokenizer.text_to_idx(comment_text)
            context_indices = tokenizer.text_to_idx(text_left + " " + text_right)
            left_indices = tokenizer.text_to_idx(text_left)
            left_with_aspect_indices = tokenizer.text_to_idx(text_left + " " + aspect)
            right_indices = tokenizer.text_to_idx(text_right)
            #right_indices = right_indices[::-1]
            right_indices = right_indices[::-1].copy()
            right_with_aspect_indices = tokenizer.text_to_idx(aspect + " " + text_right)
            #right_with_aspect_indices = right_with_aspect_indices[::-1]
            right_with_aspect_indices = right_with_aspect_indices[::-1].copy()
            aspect_indices = tokenizer.text_to_idx(aspect)
            left_len = np.sum(left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
            polarity = DICT_POLARITY[polarity.lower()]
            
            text_len = np.sum(text_indices != 0)
            bert_text = '[CLS] ' + text_left + ' ' + aspect + ' ' + text_right + ' [SEP] ' + aspect + ' [SEP]'
            concat_bert_indices = tokenizer.text_to_idx(bert_text)
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

            text_bert_indices = tokenizer.text_to_idx("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_idx("[CLS] " + aspect + " [SEP]")
            
            #dependency_graph = graph.get_graph(fname,parser) #Dependency graph corresponding to the sentence
            #constituency_tree = {} #Constituency tree corresponding to the sentence

            data = {
                # The index of the sentence. 0,1,2,3,4,5,6,7,8,9
                'idx': i // 3,
                # The index of the words in the sentence. [ 1996  9019  2003  2205 13554  1012     0     0     0     0].
                'text_indices': text_indices,
                # The index of the words in the sentence without aspect words。[ 1996  2003  2205 13554  1012     0     0     0     0     0]
                'context_indices': context_indices,
                # The index of the words to the left of aspect in the sentence(without aspect words).[ 1996  0     0     0     0     0     0     0     0     0]
                'left_indices': left_indices,
                # The index of the words to the left of aspect in the sentence(with aspect words).[ 1996  9019     0     0     0     0     0     0     0     0]
                'left_with_aspect_indices': left_with_aspect_indices,
                # The index of the words to the right of aspect in the sentence(without aspect words).[2003  2205 13554  1012     0     0     0     0     0     0].
                'right_indices': right_indices,
                # The index of the words to the right of aspect in the sentence(with aspect words).[9019  2003  2205 13554  1012     0     0     0     0     0].
                'right_with_aspect_indices': right_with_aspect_indices,
                # The index of the aspect words in the sentence.[9019     0     0     0     0     0     0     0     0     0].
                'aspect_indices': aspect_indices,
                # The boundaries of 'aspect' in the sentence, from the starting to ending index. [1,1](np.array)
                'aspect_boundary': aspect_boundary,
                # The polarity of the sentence. 0: negative, 1: neutral, 2: positive
                'polarity': polarity,
                # The index of the words in '[CLS] The keyboard is too slick . [SEP] keyboard [SEP]'.  [  101  1996  9019  2003  2205 13554  1012   102  9019   102]
                'concat_bert_indices': concat_bert_indices,
                # Segment token indices to indicate first and second portions of the inputs. [  0  0  0  0  0 0  0   0  1   1]
                'concat_segments_indices': concat_segments_indices,
                # The index of the words in '[CLS] The keyboard is too slick . [SEP]'.  [  101  1996  9019  2003  2205 13554  1012   102]
                'text_bert_indices': text_bert_indices,
                # The index of the words in '[CLS] keyboard [SEP]'.  [  101  9019   102]
                'aspect_bert_indices': aspect_bert_indices,
                'bert_text': bert_text,
                # The original comment sentence. 'The [B-ASP]keyboard[E-ASP] is too slick .' Suppose max_seq_len = 10, padding='post', truncating='post', mode = bert
                'comment_text':comment_text
                #'dependency_graph': dependency_graph,
                #'constituency_tree':constituency_tree
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]
        # debug codes for negative strides
        # data = self.data[index]

        # # Check for negative strides
        # for key, value in data.items():
        #     if isinstance(value, np.ndarray) and any(s < 0 for s in value.strides):
        #         print(f"Negative stride detected in {key} at index {index}")

        # return data


    def __len__(self):
        return len(self.data)