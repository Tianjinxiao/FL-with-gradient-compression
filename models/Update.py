#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, batch_size=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().cuda()
        self.selected_clients = []
        self.ldr_train = torch.utils.data.DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True, num_workers= 4)


    def train(self, net, lr, ep, local_iter):
        net.train()
        # train and update
        #self.args.lr *= 0.999
        #if self.args.lr < 0.000001:
        #    self.args.lr = 0.000001
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)




        epoch_loss = []
        for iter in range(ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_idx >= local_iter:
                    # if (batch_idx % 6) != batch_num:
                    break
                #images = images.cuda()
                #labels = labels.cuda()
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                # if self.args.verbose and batch_idx % 10 == 0:
                #    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

