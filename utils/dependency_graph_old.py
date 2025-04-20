# -*- coding: utf-8 -*-
# file: dependency_graph.py
# time: 2024/9/19 1602
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause
import spacy
import dgl
import os
import pickle
import numpy as np
from tqdm import tqdm
from utils.config import DEPENDENCY_TYPES,DATASET_FILES,POS_TAGS


class SpacyParserGraph:
    """Use spacy dependency parser to create DGL graph.
    
    The class uses spacy's dependent syntax analyzer to parse sentences and constructs a graph data structure based on the parsing results.
    The resulting graph can be used for subsequent graph neural network processing.

    attributes:
        model_name(str): spacy language model
        graph: DGL graphs

    methods:
        parse_sent: Parsing input sentence
        get_graph: Returns the constructed graph
    """
    def __init__(self,dataset_name:list,sentences,tokenizer,model_name:str="en_core_web_trf",undirected:bool=True):
        """
        Initializes the SpacyParserGraph class. Load spacy's English language model.
        Args:
            dataset_name(list): The dataset to be processed. The type is a list, such as ['14lap', 'train'], 
                           where the first element represent the dataset names and 
                           the second element indicate which data('train','test','valid') is to be processed
            sentences(List(str)): All sentences contained in the dataset to be processed.
            model_name(str): A language model for spacy.
            undirected(bool): Whether the generated dependency graph is an undirected graph or a directed graph. Default is a undirected graph.
        """
        self.dataset_name = dataset_name
        self.df = sentences
        self.tokenizer = tokenizer
        self.undirected = undirected
        
        self.nlp = spacy.load(model_name)
        self.graphs = None
        
        self.graph_file = DATASET_FILES[dataset[0]][dataset[1]] + '.spacy.graph'
        

    def parse_sent(self, sentence):
        """
        Applies spacy dependency parser on text to generate graph representation.

        param:
            sentence (str): Sentence or paragraph to be parsed.
        return:
            graph representation and node/edge attributes
        """
        doc = self.nlp(sentence)
        text_indices = self.tokenizer.text_to_sequence(sentence)
        
        # Create nodes and edges
        src_nodes = []
        dst_nodes = []
        edge_type_id = []
        node_pos = []

        for token in doc:
            node_pos.append(token.pos_)
            src_nodes.append(token.head.i)
            dst_nodes.append(token.i)
            edge_type_id.append(DEPENDENCY_TYPES[token.dep_])

        return {'src_nodes': src_nodes, 'dst_nodes': dst_nodes, 'len': len(doc),
                'edge_type_id': edge_type_id, 'node_idx':text_indices, 'node_pos':node_pos}
        
    def build_graph(self):
        """ 
        converts raw text to graph and save it to a file.
        return: status
        """
        if(os.path.exists(self.graph_file)):
            print('loading spacy graph:', self.graph_file)
            self.graphs = pickle.load(open(self.graph_file, 'rb'))
        else:
            print('building dependency graph by spacy...')
            for _,text in tqdm(enumerate(self.df),total=len(self.df),desc='Processing data'):
                graph_rep = self.parse_sent(text)
                dgl_graph = dgl.graph((graph_rep['src_nodes'],graph_rep['dst_nodes']),num_nodes=graph_rep['len'])
                dgl_graph.ndata['word'] = graph_rep['node_idx']
                dgl_graph.ndata['pos'] = graph_rep['node_pos']
                dgl_graph.edata['type'] = graph_rep['edge_type_id']
                if(self.undirected):
                    dgl_graph = dgl.to_bidirected(dgl_graph)
                self.graphs.append(dgl_graph) 
        
        return self.graphs


    def get_graph(self):
        """
        Return the constructed graph.
        return:
            dgl.DGLGraph: the dependency graph
        """
        return self.graphs
    
    def save_graph(self):
        """
        Save the graph to a file.
        """
        pickle.dump(self.graphs,open(self.graph_file,'wb'))

    def __call__(self):
        self.build_graph()
        return self.get_graph()


    
if __name__ == '__main__':
    dataset = ['14lap','train']
    sentences=[
        'I charge it at night and skip taking the cord with me because of the good battery life .',
        'The tech guy then said the service center does not do 1-to-1 exchange and I have to direct my concern to the `` sales `` team , which is the retail shop which I bought my netbook from .',
        'Easy to start up and does not overheat as much as other laptops .',
        'I even got my teenage son one , because of the features that it offers , like , iChat , Photobooth , garage band and more !',
        'One night I turned the freaking thing off after using it , the next day I turn it on , no GUI , screen all dark , power light steady , hard drive light steady and not flashing as it usually does .'
    ]
    
    graph = SpacyParserGraph(dataset, sentences)
    G = graph()

    
    import networkx as nx
    import matplotlib.pyplot as plt
    

    pos = nx.kamada_kawai_layout(G)
    node_colors = ['gray'] * len(G.nodes)
    edge_color = ['gray'] * len(G.edges)
    node_labels = nx.get_node_attributes(G, 'text')
    nx.draw_networkx(
        G, pos, node_size=30, labels=node_labels, font_size=7,
        node_color=node_colors, font_color='purple', edge_color=edge_color)
    plt.show()
    plt.savefig('test.png')
