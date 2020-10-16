
import torch
from utils import *
import numpy as np
#from evaluation import *







def BatchIterator(model, phase,
        Data_loader,
        criterion,
        optimizer,
        device):


    # --------------------  Initial paprameterd
    grad_clip = 0.5  # clip gradients at an absolute value of

    print_freq = 1000
    running_loss = 0.0

    
    for i, data in enumerate(Data_loader):


        imgs, labels, _ = data

        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:

            model.eval()
            with torch.no_grad():
                outputs = model(imgs)


        loss = criterion(outputs, labels)

        if phase == 'train':

            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights

        running_loss += loss * batch_size
        if (i % 200 == 0):
            print(str(i * batch_size))




    return running_loss
