# -*- coding: utf-8 -*-
# file: argument_utils.py
# time: 2024/9/21 1530
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import argparse

def get_parameter():
    parser = argparse.ArgumentParser()

    #general parameters
    parser.add_argument('--model_name', default='gat_bert', type=str)
    parser.add_argument('--device', type=str, default='cuda',help='e.g. cuda:0')
    parser.add_argument('--max_seq_len', default=85, type=int)
    #parser.add_argument('--lower', default=True, type=bool)

    #dataset parameters
    parser.add_argument('--dataset', default='14res', type=str, help='14lap,14res,15res,16res,acl_twitter, mams')
    parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')

    #classic model parameters
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    
    #Pre-trained model parameters
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--bert_dim', default=768, type=int)

    #dependency graph parameters
    parser.add_argument('--dependency_parsers', default=['stanza'], type=list, help='Specifies the dependency tree parsers for list types. Currently, five dependency tree parsers are provided: stanza, spacy, udpipe, AllenNLP, and supar.') 
    parser.add_argument('--undirected', default=False, type=bool, help='Convert the graph to undirected graph.')
    parser.add_argument('--graph_merge', default=0, type=int, help='The method for merging multiple dependency graphs, where 0 indicates no merging.')
    parser.add_argument('--dependency_parser', default='stanza', type=str, help='If the graphs are not merged, select which parser to use to obtain the dependency graph.')
    
    #model parameters
    parser.add_argument('--gat_input_dim', type=int, default=768, help='The node embedding dimension in the input graph of GAT.')
    parser.add_argument('--gat_hidden_dim', type=int, default=200, help='GAT hidden dim.')
    parser.add_argument('--gat_output_dim', type=int, default=200, help='GAT output dim.')
    parser.add_argument('--attn_head', type=int, default=2, help='NNumber of heads in Multi-Head Attention.')
    parser.add_argument('--num_gat_layer', type=int, default=2, help='Num of layers in GAT including input and output layers')
    parser.add_argument('--feature_dropout', type=float, default=0.1, help='Dropout rate on feature.')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Dropout rate on attention weight.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    #training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Num of epochs to train. Try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--patience', type=int, default=5, help='Patience')
    parser.add_argument('--seed', default=1210, type=int, help='set seed for reproducibility')

    # other parameters
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')

    args = parser.parse_args()


    return args