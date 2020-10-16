import time
import csv
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import datetime
import torch.optim
import torch.utils.data
from torchvision import  models
from torch import nn
import torch
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from dataset import NIH
from utils import *
from batchiterator import *
from tqdm import tqdm
from ResNetModel import *
import random
import numpy as np


def ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device,LR):



    # Training parameters
    batch_size = 32

    workers = 12  # mean: how many subprocesses to use for data loading.
    N_LABELS = 14
    start_epoch = 0
    num_epochs = 64  # number of epochs to train for (if early stopping is not triggered)

    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("Validation_df path",val_df_size)

    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df path", train_df_size)

    random_seed = 33 #random.randint(0,100)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)




    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        NIH(train_df, path_image=path_image, transform=transforms.Compose([
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.RandomRotation(10),
                                                                    transforms.Scale(256),
                                                                    transforms.CenterCrop(256),
                                                                    transforms.ToTensor(),
                                                                    normalize
                                                                ])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        NIH(val_df,path_image=path_image, transform=transforms.Compose([
                                                                transforms.Scale(256),
                                                                transforms.CenterCrop(256),
                                                                transforms.ToTensor(),
                                                                normalize
                                                            ])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    if ModelType == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features


        model.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())
    
    if ModelType == 'ResNet50':
        model = ResNet50NN()

    if ModelType == 'ResNet34':
        model = ResNet34NN()

    if ModelType == 'ResNet18':
        model = ResNet18NN()
        

    if ModelType == 'Resume':
        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']



    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs')
        model = nn.DataParallel(model)

    model = model.to(device)
    
    if CriterionType == 'BCELoss':
        criterion = nn.BCELoss().to(device)

    epoch_losses_train = []
    epoch_losses_val = []

    since = time.time()

    best_loss = 999999
    best_epoch = -1
#--------------------------Start of epoch loop
    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
# -------------------------- Start of phase

        phase = 'train'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_train = running_loss / train_df_size
        epoch_losses_train.append(epoch_loss_train.item())
        print("Train_losses:", epoch_losses_train)

        phase = 'val'
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        running_loss = BatchIterator(model=model, phase=phase, Data_loader=val_loader, criterion=criterion, optimizer=optimizer, device=device)
        epoch_loss_val = running_loss / val_df_size
        epoch_losses_val.append(epoch_loss_val.item())
        print("Validation_losses:", epoch_losses_val)

        # checkpoint model if has best val loss yet
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            best_epoch = epoch
            checkpoint(model, best_loss, best_epoch, LR)

                # log training and validation loss over each epoch
        with open("results/log_train", 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            if (epoch == 1):
                logwriter.writerow(["epoch", "train_loss", "val_loss","Seed","LR"])
            logwriter.writerow([epoch, epoch_loss_train, epoch_loss_val,random_seed, LR])
# -------------------------- End of phase

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            if epoch_loss_val > best_loss:
                print("decay loss from " + str(LR) + " to " + str(LR / 2) + " as not seeing improvement in val loss")
                LR = LR / 2
                print("created new optimizer with LR " + str(LR))
                if ((epoch - best_epoch) >= 10):
                    print("no improvement in 10 epochs, break")
                    break
        #old_epoch = epoch 
    #------------------------- End of epoch loop
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    Saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, batch_size)
    #
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    best_epoch = checkpoint_best['best_epoch']
    print(best_epoch)



    return model, best_epoch


