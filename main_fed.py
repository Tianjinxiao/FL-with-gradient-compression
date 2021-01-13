#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pandas as pd
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed




if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = 'True'
    #torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)



    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.RandomCrop(32, 4),transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    #CNNCifar = CNNCifar(args=args).cuda()
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
        #net_glob = nn.parallel.DistributedDataParallel(CNNCifar, device_ids=[0, 1, 2, 3])
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    #net_glob.cuda()
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    lr = args.lr
    bs = args.local_bs
    ep = args.local_ep
    local_iter = 10
    batch_num = 0
    X_train = np.zeros(5000)
    X_test = np.zeros(5000)
    error = []


    for iter in range(args.epochs):

        if iter < 150:
            # ep = 1
            # local_iter = 10
            # bs = args.local_bs
            lr = args.lr
        elif iter < 300:
            lr = args.lr / 2
        elif iter < 450:
            lr = args.lr / 2.5
        elif iter < 600:
            lr = args.lr / 5
        elif iter < 750:
            lr = args.lr / 10
        elif iter < 900:
            lr = args.lr / 20
        elif iter < 1050:
            lr = args.lr / 25
        else:
            lr = args.lr / 50


        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], batch_size=bs)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), lr=lr, ep=ep, local_iter = local_iter)
            w1 = net_glob.state_dict()
            w2 = copy.deepcopy(w1)
            com_level = 50
            # top k compression

            for i in w.keys():
                e = w[i] - w2[i]
                ee = e.reshape(-1)

                if ee.numel() > 10000:
                    topk, a = torch.abs(ee).topk(int(ee.numel() / com_level))
                    re_e = torch.zeros(ee.numel())
                    re_e = re_e.to(args.device)
                    ee = ee.float()
                    re_e[a] = ee[a]
                    re_ee = re_e.reshape(e.size())
                    w[i] = re_ee + w2[i]

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        #print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)


        if iter %1 == 0:
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print(iter)
            print("Testing accuracy: {:.2f}".format(acc_test))
            X_test[iter//1] = acc_test
        if iter % 1 == 0:
            acc_train, abc = test_img(net_glob, dataset_train, args)
            print("Training accuracy: {:.2f}".format(acc_train))
            X_train[iter // 1] = acc_train

        #lr *= 0.99





            #bs = args.local_bs * 5

            #bs = args.local_bs * 5
            #lr += 0.0205
        #if iter > 1000:
        #    bs = args.local_bs * 5
        #    lr = 0.02
            #ep = args.local_ep * 16
        #elif iter > 1500:
        #    lr = 0.002



        time_over = time.time()
        #print('time:', time_over - time_star)

    # plot loss curve
    pd_data1 = pd.DataFrame(X_train)
    s1 = 'DELRtrain_acc{0}'.format(2)
    pd_data1.to_csv(s1)

    pd_data2 = pd.DataFrame(X_test)
    s2 = 'DELRtest_acc{0}'.format(2)
    pd_data2.to_csv(s2)



    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

