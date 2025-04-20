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

class GATBERT(nn.Module):
    def __init__(self,bert, tokenizer, args):
        super(GATBERT, self).__init__()
        
        # BERT encoder
        #self.bert = BertModel.from_pretrained('hf-mirror.com/models/bert-base-uncased')
        self.bert = BertModel.from_pretrained(
        'bert-base-uncased',
        mirror="https://mirrors.aliyun.com/huggingface-models/"     #使用阿里源加载bert模型
        )
        self.tokenizer = tokenizer      #分词器
        self.device = args.device       #使用的CUDA设备
        # GATv2 layers
        self.gat_layers = nn.ModuleList()   #GAT层
        # First GAT layer
        self.gat_layers.append(     
            dglnn.GATv2Conv(    #使用GATV2卷积层
                in_feats=args.gat_input_dim,
                out_feats=args.gat_hidden_dim,
                num_heads=args.attn_head,
                feat_drop=args.feature_dropout,
                attn_drop=args.attn_drop,
                negative_slope=args.alpha,
                allow_zero_in_degree=True  # 允许零入度节点
            )
        )
        # Additional GAT layers     构建一个包含多个 GATv2Conv 层的图神经网络
        for _ in range(args.num_gat_layer - 2):     #减去2层：第一层和最后一层
            self.gat_layers.append(
                dglnn.GATv2Conv(
                    in_feats=args.gat_hidden_dim * args.attn_head,      #因为每一层的输出通常是多头注意力的结果，所以输入特征的维度是上一层输出的维度乘注意力头数量。
                    out_feats=args.gat_hidden_dim,
                    num_heads=args.attn_head,
                    feat_drop=args.feature_dropout,
                    attn_drop=args.attn_drop,
                    negative_slope=args.alpha,
                allow_zero_in_degree=True  # 允许零入度节点
                )
            )
            
        # Output of GAT layer       输出层-最后一层使用单头，因为最后一层的主要目标是生成最终的节点嵌入或图嵌入，而不是进一步增强表达能力     
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
        self.classifier = nn.Sequential(        #分类器，用于将输入特征映射到最终的类别空间
            nn.Linear(args.gat_output_dim+args.bert_dim, args.gat_output_dim),      #输入维度由args.gat_output_dim+args.bert_dim组成
            nn.Linear(args.gat_output_dim, args.gat_output_dim),
            nn.ReLU(),      #缓解梯度消失问题
            nn.Dropout(args.dropout),       #丢弃部分神经元，防止过拟合
            nn.Linear(args.gat_output_dim, args.num_class),     #最终输出维度args.num_class
        )
    @staticmethod
    # 因为 BERT 的分词方式可能会将一个单词拆分成多个子词，而依赖树图中的节点通常是完整的单词，因此需要通过这种映射方法将两者对齐。
    def find_ordered_mappings(list1, list2):        #list1:依赖图中的节点索引；list2:token索引
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

    def _get_embeddings(self,text_bert_indices,last_hidden_states,depend_graph):        #aspect_indices: 方面词的索引;text_bert_indices: BERT 的 tokenized 输入序列;bert_segments_ids: BERT 的句子分割标记;depend_graphs: 依赖树图的列表，每个图对应一个句子
        batch_size = last_hidden_states.shape[0]        #获取batch中的句子及其依赖图数量
        node_feats_embedding = []
        for i in range(batch_size):
            word_idxs_in_graph = list(depend_graph[i].ndata['word'])        #word_idxs_in_graph: 当前依赖树图中所有节点的单词索引
            word_idxs_in_bert = list(text_bert_indices[i])            #word_idxs_in_bert: 当前句子在bert中的索引
            corr_mapping = self.find_ordered_mappings(word_idxs_in_graph,word_idxs_in_bert)
            token_embeddings = []
            for graph_idx,bert_idx in corr_mapping:
                if isinstance(bert_idx,list):       #如果bert_idx是一个list，说明bert_idx是一个范围，需要求平均
                    token_embeddings.append(torch.mean(last_hidden_states[i][bert_idx],dim=0))      #对范围内的所有token的embedding求平均
                else:
                    token_embeddings.append(last_hidden_states[i][bert_idx])        #如果bert_idx是一个数字，说明bert_idx是一个索引，直接取对应的token的embedding
                    
            # 将 token_embeddings 转换为张量并存储在图的节点数据中
            depend_graph[i].ndata['embedding'] = torch.stack(token_embeddings)

            # 将图的节点嵌入表示添加到 node_feats_embedding 中
            node_feats_embedding.append(depend_graph[i].ndata['embedding'])
            #     depend_graph[i].ndata['embedding'] = torch.tensor(token_embeddings)
            # node_feats_embedding.append(torch.tensor(token_embeddings))
        return node_feats_embedding
    @staticmethod
    def _find_aspect_indices(depend_graph, aspect_indices):     #depengd graph依赖图 aspect_indices方面词索引
        aspect_idx_graph = []       #aspect_idx_graph方面词在依赖图中的索引
        for idx in aspect_indices:
            if idx != 0:
                indices = (depend_graph.ndata['word'] == idx).nonzero(as_tuple=True)[0]     #返回节点单词索引等于方面词索引时的索引
                aspect_idx_graph.extend(indices.tolist())       #将张量转换成列表并添加到aspect_idx_graph中
        return aspect_idx_graph    


    def forward(self, inputs,depend_graphs):        #核心代码：前向传播函数
        """
        Args:
            g: DGLGraph - The dependency tree graph
            input_ids: Tensor - BERT input ids
            attention_mask: Tensor - BERT attention mask
            aspect_indices: Tensor - Indices of aspect terms in the sentence
        """
        # Get BERT embeddings
        aspect_indices, text_bert_indices, bert_segments_ids=inputs[0],inputs[1],inputs[2]      #aspect_indices: 方面词的索引 text_bert_indices: token索引 bert_segments_ids: Bert句子分割标记
        attention_mask = [[1 if token != 0 else 0 for token in seq] for seq in text_bert_indices]
        attention_mask = torch.tensor(attention_mask).to(self.device)       #将attention_mask转为张量并传到GPU
        
        with torch.no_grad():       #禁用梯度计算，因为 BERT 是预训练模型，不需要更新其参数。
            bert_outputs = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        last_hidden_states = bert_outputs[0]
        pooled_output = bert_outputs[1]     #看一下是不是指方面词，不是句子

        # 使用 BERT 输出的 token 隐藏层作为 DGL 的节点输入
        h = self._get_embeddings(text_bert_indices, last_hidden_states, depend_graphs)

        # 合并多个图为一个大图
        batched_graph = dgl.batch(depend_graphs)
        batched_h = torch.cat(h, dim=0)

        # Process through GAT layers
        for j, gat_layer in enumerate(self.gat_layers):
            batched_h = gat_layer(batched_graph, batched_h)
            if j < len(self.gat_layers) - 1:  # 不是最后一层时，将多头注意力的结果展平，以适应下一层的输入维度
                batched_h = batched_h.view(batched_h.shape[0], -1)

        batched_h = batched_h.squeeze(1)        #去除多余的维度
        
        logits = []

        # 拆分大图为单个图
        unbatched_graphs = dgl.unbatch(batched_graph)       #拆分
        node_offset = 0
        for i, single_graph in enumerate(unbatched_graphs):
            num_nodes = single_graph.num_nodes()        #获取图中有多少个词
            g_result = batched_h[node_offset:node_offset + num_nodes]       #获取单个图中节点的范围
            node_offset += num_nodes

            aspect_idx_graph = self._find_aspect_indices(single_graph, aspect_indices[i])       #找到当前图中与方面词对应的节点索引

            # Get aspect term representations
            aspect_feats = g_result[aspect_idx_graph]       #从当前图的节点嵌入中提取方面词的特征
            aspect_feats = torch.mean(aspect_feats, dim=0, keepdim=True)        #对所有方面词的嵌入取平均值，得到一个单一的特征向量     因为一个句子有多个方面词，所以需要平均
            combined_feats = torch.cat((aspect_feats, pooled_output[i].unsqueeze(0)), dim=1)    

            # Classification
            logit = self.classifier(combined_feats)     #得到当前样本的预测概率分布
            logits.append(logit)
        logits = torch.stack(logits).squeeze(1)     #将所有样本的分类结果堆叠成一个张量并且去除多余的维度
        return logits
