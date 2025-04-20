# -*- coding: utf-8 -*-
# file: __init__.py
# time: 2024/9/18 1550
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from transformers import BertModel

class GATBERT_old(nn.Module):
    def __init__(self,bert, args):
        super(GATBERT, self).__init__()
        
        # BERT encoder
        self.bert = bert
        self.device = args.device
        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        # First GAT layer
        self.gat_layers.append(
            dglnn.GATv2Conv(
                in_feats=args.gat_input_dim,
                out_feats=args.gat_hidden_dim,
                num_heads=args.attn_head,
                feat_drop=args.feature_dropout,
                attn_drop=args.attn_drop,
                negative_slope=args.alpha,
                allow_zero_in_degree=True  # 允许零入度节点
            )
        )
        # Additional GAT layers
        for _ in range(args.num_gat_layer - 2):
            self.gat_layers.append(
                dglnn.GATv2Conv(
                    in_feats=args.gat_hidden_dim * args.attn_head,
                    out_feats=args.gat_hidden_dim,
                    num_heads=args.attn_head,
                    feat_drop=args.feature_dropout,
                    attn_drop=args.attn_drop,
                    negative_slope=args.alpha,
                allow_zero_in_degree=True  # 允许零入度节点
                )
            )
            
        # Output of GAT layer
        if args.num_gat_layer > 1:
            self.gat_layers.append(
                dglnn.GATv2Conv(
                    in_feats=args.gat_hidden_dim * args.attn_head,
                    out_feats=args.gat_output_dim, 
                    num_heads=1,  # 输出层通常使用单头
                    feat_drop=args.feature_dropout,
                    attn_drop=args.attn_drop,
                allow_zero_in_degree=True  # 允许零入度节点
                )
            )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(args.gat_output_dim+args.bert_dim, args.gat_output_dim),
            nn.Linear(args.gat_output_dim, args.gat_output_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.gat_output_dim, args.gat_output_dim),
        )

        #tesf for bert
        # self.dropout = nn.Dropout(args.dropout)
        # self.dense = nn.Linear(args.bert_dim, args.num_class)
    @staticmethod
    def find_ordered_mappings(list1, list2):
        """
        Map the relationship between two lists while maintaining the original order and corresponding spans.
        查找两个列表的映射关系，保持原始顺序和跨度对应    
        Args:
            list1 (list): The first list
            list2 (list): The second list
        
        Returns:
            list: List of element mapping relationships 元素映射关系的列表
        """
        mappings = []
        list2_len = len(list2)
        current_list2_index = 0

        for list1_index, val1 in enumerate(list1):
            for j in range(current_list2_index, list2_len):
                if list2[j] == val1:
                    mappings.append((list1_index, j))
                    current_list2_index = j + 1
                    break
            else:
                mappings.append((list1_index,-1))

        list1_len = len(mappings)
        for i in range(list1_len):
            if mappings[i][1] == -1:
                lst1_idx =mappings[i][0]
                
                last_lst2idx = 0            
                for k in range(i-1,-1,-1):
                    if isinstance(mappings[k][1],list):
                        last_lst2idx = mappings[k][1][-1]
                    else:
                        last_lst2idx = mappings[k][1]
                    if last_lst2idx!=-1:
                        break
                    
                forward_lst2idx = list1_len-1
                for k in range(i+1, list1_len):
                    if isinstance(mappings[k][1],list):
                        forward_lst2idx = mappings[k][1][-1]
                    else:
                        forward_lst2idx = mappings[k][1]
                    if forward_lst2idx!=-1:
                        break
                    
                if last_lst2idx+1 >= forward_lst2idx-1:
                    mappings[i] =(lst1_idx,forward_lst2idx-1)
                else:                
                    mappings[i] =(lst1_idx,[last_lst2idx+1,forward_lst2idx-1])               

        return mappings

    def _get_embeddings(self,text_bert_indices,last_hidden_states,depend_graph):
        batch_size = last_hidden_states.shape[0]
        node_feats_embedding = []
        for i in range(batch_size):
            word_idxs_in_graph = list(depend_graph[i].ndata['word'])
            word_idxs_in_bert = list(text_bert_indices[i])
            corr_mapping = self.find_ordered_mappings(word_idxs_in_graph,word_idxs_in_bert)
            token_embeddings = []
            for graph_idx,bert_idx in corr_mapping:
                if isinstance(bert_idx,list):
                    token_embeddings.append(torch.mean(last_hidden_states[i][bert_idx],dim=0))
                else:
                    token_embeddings.append(last_hidden_states[i][bert_idx])
                    
            # 将 token_embeddings 转换为张量并存储在图的节点数据中
            depend_graph[i].ndata['embedding'] = torch.stack(token_embeddings)

            # 将图的节点嵌入表示添加到 node_feats_embedding 中
            node_feats_embedding.append(depend_graph[i].ndata['embedding'])
            #     depend_graph[i].ndata['embedding'] = torch.tensor(token_embeddings)
            # node_feats_embedding.append(torch.tensor(token_embeddings))
        return node_feats_embedding
    @staticmethod
    def _find_aspect_indices(depend_graph, aspect_indices):
        aspect_idx_graph = []
        for idx in aspect_indices:
            if idx != 0:
                indices = (depend_graph.ndata['word'] == idx).nonzero(as_tuple=True)[0]
                aspect_idx_graph.extend(indices.tolist())
        return aspect_idx_graph
    def forward(self, inputs,depend_graphs):
        """
        Args:
            g: DGLGraph - The dependency tree graph
            input_ids: Tensor - BERT input ids
            attention_mask: Tensor - BERT attention mask
            aspect_indices: Tensor - Indices of aspect terms in the sentence
        """
        # Get BERT embeddings
        aspect_indices, text_bert_indices, bert_segments_ids=inputs[0],inputs[1],inputs[2]
        attention_mask = [[1 if token != 0 else 0 for token in seq] for seq in text_bert_indices]
        attention_mask = torch.tensor(attention_mask).to(self.device)
        
        with torch.no_grad():
            bert_outputs = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        last_hidden_states = bert_outputs[0]
        pooled_output = bert_outputs[1]     
        
        logits = []

        # Process through GAT layers
        h = self._get_embeddings(text_bert_indices,last_hidden_states,depend_graphs)

        for i in range(len(depend_graphs)):
            g_result = h[i]

            for j, gat_layer in enumerate(self.gat_layers):
                g_result = gat_layer(depend_graphs[i], g_result)
                if j < len(self.gat_layers) - 1:  # 不是最后一层时进行维度改变
                    g_result = g_result.view(g_result.shape[0], -1)
                
            
            g_result = g_result.squeeze(1)
            aspect_idx_graph=self._find_aspect_indices(depend_graphs[i], aspect_indices[i])

            # Get aspect term representations
            aspect_feats = g_result[aspect_idx_graph]
            aspect_feats = torch.mean(aspect_feats, dim=0, keepdim=True)
            combined_feats = torch.cat((aspect_feats, pooled_output[i].unsqueeze(0)), dim=1)

            # Classification
            #logit = self.classifier(aspect_feats)
            logit = self.classifier(combined_feats)
            logits.append(logit)
        
        logits = torch.stack(logits).squeeze(1)
        #logits = self.dense(self.dropout(pooled_output))

        return logits
