# -*- coding: utf-8 -*-
# file: train.py
# time: 2024/9/18 1550
# author: Feng, Yi <jmyf@21cn.com> (易锋)
# github: https://github.com/zsrainbow/
# BSD 3 clause

import math
import torch
import os
import torch.nn as nn
from sklearn import metrics

from torch.utils.data import DataLoader, random_split
from utils.tokenizer import Tokenizer4Bert,build_tokenizer,build_embedding_matrix
from transformers import BertModel
from utils.vocab import Vocab
from utils.config import DATASET_FILES
from utils.dependency_graph import DependencyGraph
from utils.data_utils import ABSADataset
from models.gat_bert import GATBERT

class Instructor:
    def __init__(self,logger,args):
        self.logger = logger
        self.args = args
        if 'bert' in args.model_name:       #bert模型
            tokenizer = Tokenizer4Bert(args.max_seq_len, args.pretrained_bert_name)     #分词
            bert = BertModel.from_pretrained(args.pretrained_bert_name)     #预训练模型
            self.model = args.model_class(bert, tokenizer,args).to(args.device)     #CUDA处理
        else:       #其他模型
            vocab = Vocab.load_vocab(args.dataset)      #加载词表
            tokenizer = build_tokenizer(        #分词
                fnames=[DATASET_FILES[args.dataset]['train'], DATASET_FILES[args.dataset]['test']], vocab=vocab,
                max_seq_len=args.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(args.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=args.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(args.embed_dim), args.dataset))     
            self.model = args.model_class(embedding_matrix, args).to(args.device)

        self.trainset = ABSADataset(args.dataset_file['train'], tokenizer)      #训练集
        self.testset = ABSADataset(args.dataset_file['test'], tokenizer)        #测试集

        ###debug
        # print(len(self.trainset))
        # train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.batch_size, shuffle=True)
        # for i_batch, batch in enumerate(train_data_loader):
        #     print(batch)            
        # for i, text in enumerate(self.trainset):
        #     print(f"Index {i}: Length {text}")

        d_graph = DependencyGraph(args.dataset,tokenizer,args.dependency_parsers,args.undirected,args.graph_merge)      #创建依赖图
        self.train_dependency_graph = d_graph.get_graph(args.dataset_file['train'],args.dependency_parser)      #训练集依赖图
        self.test_dependency_graph = d_graph.get_graph(args.dataset_file['test'],args.dependency_parser)        #测试集依赖图

        assert 0 <= args.valset_ratio < 1
        if args.valset_ratio > 0:
            valset_len = int(len(self.trainset) * args.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
            self.valid_dependency_graph = d_graph.get_graph(args.dataset_file['valid'],args.dependency_parser)
        else:
            self.valset = self.testset
            self.valid_dependency_graph = self.test_dependency_graph

        if args.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=args.device.index)))
        self._print_args()

    def _print_args(self):      #打印参数
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def _reset_params(self):        #重置参数
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):     #核心代码
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.args.epochs):     #代数
            self.model.train()
            self.logger.info('>' * 100)
            self.logger.info('epoch: {}'.format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()      #批次训练
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.args.device) for col in self.args.inputs_cols]
                
                #dependency_graph = self.train_dependency_graph[batch['idx'].tolist()].to(self.args.device)
                # 获取对应的图数据
                idx_list = batch['idx'].tolist()
                dependency_graphs = [self.train_dependency_graph[idx].to(self.args.device) for idx in idx_list]
                #dependency_graphs = torch.stack(dependency_graphs)  # 将图数据堆叠成一个张量

                inputs.append(dependency_graphs)
                outputs = self.model(inputs,dependency_graphs)
                #predicted_classes = torch.argmax(outputs, dim=1)

                targets = batch['polarity'].to(self.args.device)

                loss = criterion(outputs, targets)  #算loss：在某一个点，3个分量
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.args.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    self.logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            self.logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.args.model_name, self.args.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                self.logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.args.patience:       #提早终止
                print('>> early stop.')
                break

        return path

    def _evaluate_acc_f1(self, data_loader):        #评估
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.args.device) for col in self.args.inputs_cols]
                t_targets = t_batch['polarity'].to(self.args.device)

                # 获取对应的图数据
                idx_list = t_batch['idx'].tolist()
                dependency_graphs = [self.train_dependency_graph[idx].to(self.args.device) for idx in idx_list]

                t_outputs = self.model(t_inputs,dependency_graphs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):      #外部接口
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()       #损失函数交叉熵
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.args.optimizer(_params, lr=self.args.lr, weight_decay=self.args.weight_decay)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.args.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.args.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.args.batch_size, shuffle=False)

        self._reset_params()    #重置模型参数
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)     #保存最好性能模型
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        self.logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))