from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import util,datas,train
from models import model


#directory for dataset
dataroot='../../../local/dataset/food-101/images'

#batchsize
batch_size=32

#image_size 
image_size=224

#number of training epochs
num_epochs=5

#learning rate
lr=0.0002

# beta1 hyperparam for adam
beta1=0.5

#tensorboard runs path
runs_path="runs/vgg_at"

#weight path
weight_path="weight/vgg_at.pth"


#make dataloader
dataset=dset.ImageFolder(root=dataroot,
                        transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                        ]))


trainset, testset = torch.utils.data.random_split(dataset , [95000, 6000])

trainloader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,
                                      shuffle=True,num_workers=2)

testloader=torch.utils.data.DataLoader(testset,batch_size=batch_size,
                                      shuffle=True,num_workers=2)


#classes
classes=datas.class_list()

#gpu activate
device=torch.device('cuda:0')

#model
att_vgg = model.at_vgg()

# send model to gpu
net=att_vgg.to(device)

#cross  entropy
criterion=nn.CrossEntropyLoss()

#adam
optimizer=optim.Adam(net.parameters(),lr=lr,betas=(beta1,0.999))

#training

train.model_train(trainloader,net,criterion,optimizer,device,num_epochs,runs_path,weight_path)
