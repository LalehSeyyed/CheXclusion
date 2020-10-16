import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.misc import imread, imresize
from PIL import Image




class CheXpert(Dataset):
    def __init__(self, dataframe, path_image, finding="any", transform=None):
        self.dataframe = dataframe
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.finding = finding
        self.transform = transform
        self.path_image = path_image

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.dataframe.columns:
                if len(self.dataframe[self.dataframe[finding] == 1]) > 0:
                    self.dataframe = self.dataframe[self.dataframe[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")
        self.PRED_LABEL = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]

        img = imread(os.path.join(self.path_image, item["Path"]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img)
        # img = imresize(img, (256, 256))
        # img = img.transpose(2, 0, 1)
        # assert img.shape == (3, 256, 256)
        # assert np.max(img) <= 255
        # img = torch.FloatTensor(img / 255.)
        if self.transform is not None:
            img = self.transform(img)

        # label = np.zeros(len(self.PRED_LABEL), dtype=int)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):

            if (self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.dataframe[self.PRED_LABEL[i].strip()].iloc[idx].astype('float')

        return img, label, item["Path"]#self.dataframe.index[idx]

    def __len__(self):
        return self.dataset_size
