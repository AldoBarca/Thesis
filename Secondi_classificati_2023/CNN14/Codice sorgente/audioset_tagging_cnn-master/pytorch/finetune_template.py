import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import utilities
from models import Cnn14,init_layer
import config


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..', 'losses')))
from torch import nn
from tqdm import tqdm
from losses import SupConLoss
from da import RandomCrop, Resize, Compander, GaussNoise, FreqShift, MixRandom
#from models import ResNet
from torchinfo import summary
from args import args
import math
import h5py
import os
#from metadata import *

from speechbrain.inference.classifiers import AudioClassifier 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'metadata')))

import config
#import metadata

class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output =  torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict


def train():

    # Arugments & parameters
    sample_rate = 22050
    window_size = 512
    hop_size =128
    mel_bins = 64 #128
    fmin = 50
    fmax = 11025
    model_type = "Transfer_Cnn14"
    pretrained_checkpoint_path = "C:/Users/aldob/Documents/GitHub/Thesis/Secondi_classificati_2023/CNN14/Codice sorgente/audioset_tagging_cnn-master/Cnn14_16k_mAP=0.438.pth"
    freeze_base = True
    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

    classes_num = 46
    pretrain = True if pretrained_checkpoint_path else False
    
    # Model
    Model = eval(model_type)
    encoder = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        classes_num, freeze_base)
    
    # Load pretrained model
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        encoder.load_from_pretrain(pretrained_checkpoint_path)

    print('GPU number: {}'.format(torch.cuda.device_count()))
   
    return encoder


def train_scl(encoder, train_loader, transform1, transform2, args):

    print(f"Training starting on {args.device}")
    
    loss_fn = SupConLoss(temperature=args.tau, device=args.device)
    
    optim = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    num_epochs = args.epochs

    ckpt_dir = os.path.join(args.traindir, '../model/')
    os.makedirs(ckpt_dir, exist_ok=True) 
    last_model_path = os.path.join(ckpt_dir, 'ckpt.pth')

    encoder = encoder.to(args.device)
    
    for epoch in range(1, num_epochs+1):
        tr_loss = 0.
        print("Epoch {}".format(epoch))
        adjust_learning_rate(optim, args.lr, epoch, num_epochs+1)
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)

            x1 = transform1(x); x2 = transform2(x)
            x1=torch.tensor(x1)
            x2=torch.tensor(x2)
            _, x_out1 = encoder(x1); _, x_out2 = encoder(x2)

            if args.method == 'ssl':
                loss = loss_fn(x_out1, x_out2)
            elif args.method == 'scl':
                loss = loss_fn(x_out1, x_out2, y)
            tr_loss += loss.item()

            loss.backward()
            optim.step()

        tr_loss = tr_loss/len(train_iterator)
        print('Average train loss: {}'.format(tr_loss))

    torch.save({'encoder':encoder.state_dict()},last_model_path)

    return encoder

def adjust_learning_rate(optimizer, init_lr, epoch, tot_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / tot_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == "__main__":

    # Load data
    hdf_tr = os.path.join(args.traindir,'train.h5')
    hdf_train = h5py.File(hdf_tr, 'r+')
    X = hdf_train['data'][:]
    Y = hdf_train['label'][:]
    print(X.shape)
    # Create dataset
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).unsqueeze(1), torch.tensor(Y.squeeze(), dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    # Data augmentation
    time_steps = int(args.sr / (1000/args.len) / args.hoplen)
    rc = RandomCrop(n_mels=args.nmels, time_steps=time_steps, tcrop_ratio=args.tratio)
    resize = Resize(n_mels=args.nmels, time_steps=time_steps)
    awgn = GaussNoise(stdev_gen=args.noise, device=args.device)
    comp = Compander(comp_alpha=args.comp)
    mix = MixRandom(device=args.device)
    fshift = FreqShift(Fshift=args.fshift)

    # Prepare views
    transform1 = nn.Sequential(mix, fshift, rc, resize, comp, awgn) # only one branch has mixing with a background sound
    transform2 = nn.Sequential(fshift, rc, resize, comp, awgn)
    
    # Prepare model

    
    #encoder = ResNet(method=args.method)  
    #encoder = AudioClassifier.from_hparams(source="speechbrain/cnn14-esc50", savedir='pretrained_models/cnn14-esc50')  
    
   


    #if args.mode == 'train':
    encoder=train()

    print(summary(encoder))

    # Launch training
    model = train_scl(encoder, train_loader, transform1, transform2, args)






    print('Load pretrained model successfully!')



  

