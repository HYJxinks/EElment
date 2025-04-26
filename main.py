import random
import numpy
import torch
import os
import sys
import logging

import matplotlib.pyplot as plt
import pickle


from time import strftime, localtime
from argument_utils import get_parameter
from utils.config import model_classes, DATASET_FILES, input_colses, initializers, optimizers
from train import Instructor

if __name__ == '__main__':
    
    '''# 设置代理环境变量
    proxy_address = 'http://172.16.61.163:808'  # 替换为你的代理地址和端口
    os.environ['HTTP_PROXY'] = proxy_address
    os.environ['HTTPS_PROXY'] = proxy_address'''

    # initialize parameters
    args = get_parameter()

    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    args.model_class = model_classes[args.model_name]
    args.dataset_file = DATASET_FILES[args.dataset]
    args.inputs_cols = input_colses[args.model_name]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device is None else torch.device(args.device)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    log_file = './log/{}-{}-{}.log'.format(args.model_name, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(logger,args)
    ins.run()
