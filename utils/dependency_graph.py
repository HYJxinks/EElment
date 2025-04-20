import dgl
import json
import torch
import numpy as np
from tqdm import tqdm
from utils.config import DEPENDENCY_TYPES,DATASET_FILES,POS_TAGS

class DependencyGraph(object):
    """
    Dependency Graph
    """
    def __init__(self, dataset_name:str,tokenizer,parsers:list=['stanza'],undirected:bool=True,lower:bool=True,graph_merge:int=0):
        """
        Initializes the Dependency Graph class. 
        Args:
            dataset_name(str): The dataset to be processed. The type is a string, such as 'twitter', '14lap' etc.
            tokenizer(Tokenizer): The tokenizer used to tokenize the sentences.
            parser(list): The parser used to parse the sentences. The type is a list, such as ['stanza','spacy'].
            undirected(bool): Whether the generated dependency graph is an undirected graph or a directed graph. Default is a undirected graph.
            graph_merge(int): 0 means no merging, 1 means merging with training data, 2 means merging with training and validation data. Default is 0.
        """
        self.parsers = parsers
        self.dataset_name = dataset_name
        self.undirected = undirected
        self.tokenizer = tokenizer
        self.lower = lower
        self.graph_merge = graph_merge
        self.dependency_graphs = {
            'train': {},
            'test': {},
            'valid': {}
        }
        self.dependency_merge_graph = {
            'train': {},
            'test': {},
            'valid': {}
        }
        
        for item in DATASET_FILES[dataset_name]:
            if item == 'train':
                for parser in parsers:
                    g_fname = DATASET_FILES[dataset_name][item] + '.' + parser + '.dep.graph'
                    self.dependency_graphs['train'][parser] = self._build_graph(DATASET_FILES[dataset_name][item],g_fname)
            elif item == 'test':
                for parser in parsers:
                    g_fname = DATASET_FILES[dataset_name][item] + '.' + parser + '.' + 'dep.graph'
                    self.dependency_graphs['test'][parser] = self._build_graph(DATASET_FILES[dataset_name][item],g_fname)
            else:
                for parser in parsers:
                    g_fname = DATASET_FILES[dataset_name][item] + '.' + parser + '.' + 'dep.graph'
                    self.dependency_graphs['valid'][parser] = self._build_graph(DATASET_FILES[dataset_name][item],g_fname)
        
    def _get_sentences(self,fname):
        """
        Loads the sentences from the dataset.
        """
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].strip()
            comment_text = text_left +" "+ aspect +" "+ text_right
            all_data.append(comment_text)
        
        return all_data

    def _build_graph(self,fname, g_fname):
        """
        Builds the dependency graph.
        """
        print(f'Building dependency graph of {fname}...')
        
        depend_graphs={}

        sentences = self._get_sentences(fname)
        
        with open(g_fname, 'r', encoding='utf-8') as f:
            graph_text = json.load(f)
            for idx_sentence,item in enumerate(graph_text):
                
                #下面是为了处理，哪个是根节点，并把根节点的编号进行还原
                root_ix = graph_text[item]['nodes_head'].index(0)
                 
                # root对应节点的编号为0                
                nodes_ix = graph_text[item]['nodes']
                nodes_ix[root_ix] = 0
                nodes_ix = [x-1 if x>root_ix else x for x in nodes_ix ]
                
                heads_ix = graph_text[item]['nodes_head']
                heads_ix = [0 if x==root_ix+1 else x for x in heads_ix]
                heads_ix = [x-1 if x>root_ix else x for x in heads_ix ]
                
                dgl_graph = dgl.graph((heads_ix,nodes_ix),num_nodes=graph_text[item]['len'],idtype=torch.int32)

                sentence = sentences[idx_sentence]
                
                if self.lower:
                    text_indices = [self.tokenizer.text_to_sequence(text.lower())[0] for text in graph_text[item]['nodes_text']]
                else:
                    text_indices = [self.tokenizer.text_to_sequence(text)[0] for text in graph_text[item]['nodes_text']]
                
                dgl_graph.ndata['word'] = torch.tensor(text_indices)
                
                # 下面的pos后面要做嵌入表示
                nodes_pos = graph_text[item]['nodes_pos']
                nodes_pos.insert(0,nodes_pos[root_ix])
                nodes_pos.pop(root_ix+1)
                nodes_posid = [POS_TAGS[pos] for pos in nodes_pos]
                dgl_graph.ndata['pos'] = torch.tensor(nodes_posid)
                
                #下面的边要做嵌入表示
                edges_type = graph_text[item]['edges_type']
                edges_typeid = [DEPENDENCY_TYPES[edge] for edge in edges_type]
                dgl_graph.edata['type'] = torch.tensor(edges_typeid)
                
                #删除根节点上行的环，主要是删除root到root的这一个环
                dgl_graph = dgl.remove_self_loop(dgl_graph)

                if(self.undirected):
                    dgl_graph = dgl.to_bidirected(dgl_graph)
                
                depend_graphs[int(item)]=dgl_graph
        return depend_graphs
    
    def get_graph(self,fname,parser='stanza'):
        if len(self.dependency_graphs['train']) == 1:
            if 'train' in fname.lower():
                return self.dependency_graphs['train'][parser]
            elif 'test' in fname.lower():
                return self.dependency_graphs['test'][parser]
            else:
                return self.dependency_graphs['valid'][parser]
        else:
            if 'train' in fname.lower():
                return self.dependency_merge_graph['train'][parser]
            elif 'test' in fname.lower():
                return self.dependency_merge_graph['test'][parser]
            else:
                return self.dependency_merge_graph['valid'][parser]
        
    
    def merge_graph_by_union(self):
        """
        Merges the dependency graphs of the training, validation and test sets.
        """
        if len(self.parsers)<=1:
            self.train_merge_graph = self.train_dependency_graph
            self.test_merge_graph = self.test_dependency_graph
            self.validate_merge_graph = self.validate_dependency_graph
            return
                
    def merge_graph(self,graphs):
        """
        Merges the dependency graphs of the training, validation and test sets.
        此处所有图的节点假设是相同的，只有边是不同的
        """
        pass
            
            