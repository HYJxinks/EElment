# -*- coding: utf-8 -*-
# file: relation_types.py
# time: 2024/9/18 1550
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import torch
from models.gat_bert import GATBERT

DICT_POLARITY = {'negative': 0, 'neutral': 1, 'positive': 2}
BERT_MODEL_NAME = 'https://hf-mirror.com/models/bert-base-uncased'  # 使用hf-mirror镜像源

POS_TAGS = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'AUX': 3, 'CONJ':4, 'CCONJ': 5, 'DET': 6, 'INTJ': 7, 'NOUN': 8, 'NUM': 9, 'PART': 10, 
            'PRON': 11, 'PROPN': 12, 'PUNCT': 13, 'SCONJ': 14, 'SYM': 15, 'VERB': 16,  'X': 17}

# DEPENDENCY_TYPES = {0:'ROOT', 1:'acl', 2:'acomp', 3:'advcl', 4:'advmod', 5:'agent', 6:'amod', 7:'appos', 8:'attr', 9:'aux',
#                       10:'auxpass', 11:'case', 12:'cc', 13:'ccomp', 14:'compound', 15:'conj', 16:'csubj', 17:'csubjpass', 18:'dative',
#                       19:'dep', 20:'det', 21:'dobj', 22:'expl', 23:'intj', 24:'mark', 25:'meta', 26:'neg', 27:'nmod', 28:'npadvmod', 29:'nsubj',
#                       30:'nsubjpass', 31:'nummod', 32:'oprd', 33:'parataxis', 34:'pcomp', 35:'pobj', 36:'poss', 37:'preconj', 38:'predet',
#                       39:'prep', 40:'prt', 41:'punct', 42:'quantmod', 43:'relcl', 44:'xcomp', 45:'next', 46:'coref', 47:'self_loop',
#                       48:'subtok'}
DEPENDENCY_TYPES = {'ROOT': 0, 'acl': 1, 'acomp': 2, 'advcl': 3, 'advmod': 4, 'agent': 5, 'amod': 6, 'appos': 7, 'attr': 8, 'aux': 9, 
                    'auxpass': 10, 'case': 11, 'cc': 12, 'ccomp': 13, 'compound': 14, 'conj': 15, 'csubj': 16, 'csubjpass': 17, 'dative': 18, 
                    'dep': 19, 'det': 20, 'dobj': 21, 'expl': 22, 'intj': 23, 'mark': 24, 'meta': 25, 'neg': 26, 'nmod': 27, 'npadvmod': 28, 
                    'nsubj': 29, 'nsubjpass': 30, 'nummod': 31, 'oprd': 32, 'parataxis': 33, 'pcomp': 34, 'pobj': 35, 'poss': 36, 'preconj': 37, 
                    'predet': 38, 'prep': 39, 'prt': 40, 'punct': 41, 'quantmod': 42, 'relcl': 43, 'xcomp': 44, 'next': 45, 'coref': 46, 
                    'self_loop': 47, 'subtok': 48, 'acl:relcl':49, 'aux:pass':10,'csubj:pass':17,'nsubj:pass':30,'root':0,
                    'cop':50,'fixed':51,'flat':52,'infmod':53,'iobj':54,'list':55,'mwe':56,'nmod:npmod':57,'nmod:poss':58,'nmod:tmod':59,
                    'nn':60,'nsubj:outer':61,'num':62,'number':63,'obj':64,'obl':65,'obl:agent':66,'obl:npmod':67,'obl:tmod':68,'orphan':69,
                    'partmod':70,'possessive':71,'rcmod':72,'goeswith':73,'dislocated':74,'discourse':75,'det:predet':76,'csubj:outer':77,
                    'compound:prt':78,'cc:preconj':79,'advcl:relcl':80,'reparandum':81,'tmod':82,'vocative':83}


# CONSTITUENCY_TYPES = {0:'S',1:'ADJP',2:'ADVP',3:'CONJP',4:'FRAG',5:'INTJ',6:'LST',7:'NAC',8:'NP',9:'NX',10:'PP',11:'PRN',
#                       12:'PRT',13:'QP', 14:'ROOT',15:'RRC',16:'SBAR',17:'SBARQ',18:'SINV',19:'SQ',20:'UCP',21:'VP',22:'WHADJP',
#                       23:'WHADVP',24:'WHNP',25:'WHPP',26:'X'}
CONSTITUENCY_TYPES = {'S': 0, 'ADJP': 1, 'ADVP': 2, 'CONJP': 3, 'FRAG': 4, 'INTJ': 5, 'LST': 6, 'NAC': 7, 'NP': 8, 'NX': 9, 'PP': 10, 
                      'PRN': 11, 'PRT': 12, 'QP': 13, 'ROOT': 14, 'RRC': 15, 'SBAR': 16, 'SBARQ': 17, 'SINV': 18, 'SQ': 19, 'UCP': 20, 
                      'VP': 21, 'WHADJP': 22, 'WHADVP': 23, 'WHNP': 24, 'WHPP': 25, 'X': 26}


DATASET_FILES = {
    'twitter': {
        'train': './datasets/acl-14-short-data/train.raw',
        'test': './datasets/acl-14-short-data/test.raw'
    },
    '14lap': {
        'train': './datasets/semeval14/Laptops_Train.xml.seg',
        'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
    },
    '14res': {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    '15res': {
        'train': './datasets/semeval15/restaurant_train.raw',
        'test': './datasets/semeval15/restaurant_test.raw'
    },
    '16res': {
        'train': './datasets/semeval16/restaurant_train.raw',
        'test': './datasets/semeval16/restaurant_test.raw'
    },
    'mams':{
        'train':'./datasets/mams/train.xml.dat',
        'test':'./datasets/mams/test.xml.dat',
        'valid':'./datasets/mams/valid.xml.dat'
    }
}

input_colses = {
    'gat_bert': ['aspect_indices','concat_bert_indices', 'concat_segments_indices'],
    'lstm': ['text_indices'],
    'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
    'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
    'atae_lstm': ['text_indices', 'aspect_indices'],
    'ian': ['text_indices', 'aspect_indices'],
    'memnet': ['context_indices', 'aspect_indices'],
    'ram': ['text_indices', 'aspect_indices', 'left_indices'],
    'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
    'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_boundary'],
    'aoa': ['text_indices', 'aspect_indices'],
    'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
    'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
    'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
    'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices']
}


initializers = {
    'xavier_uniform_': torch.nn.init.xavier_uniform_,
    'xavier_normal_': torch.nn.init.xavier_normal_,
    'orthogonal_': torch.nn.init.orthogonal_,
}

optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD,
}

model_classes = {
    'gat_bert':GATBERT,
    # 'lstm': LSTM,
    # 'td_lstm': TD_LSTM,
    # 'tc_lstm': TC_LSTM,
    # 'atae_lstm': ATAE_LSTM,
    # 'ian': IAN,
    # 'memnet': MemNet,
    # 'ram': RAM,
    # 'cabasc': Cabasc,
    # 'tnet_lf': TNet_LF,
    # 'aoa': AOA,
    # 'mgan': MGAN,
    # 'asgcn': ASGCN,
    # 'bert_spc': BERT_SPC,
    # 'aen_bert': AEN_BERT,
    # 'lcf_bert': LCF_BERT,
}
