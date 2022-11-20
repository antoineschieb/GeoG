import torch
from os import listdir
from os.path import isfile, join
import cv2
from sklearn.preprocessing import OneHotEncoder
from patchify import patchify
import numpy as np
from torch.utils.data import Dataset


class DS_1080p_Country(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [f.split(".png")[0] for f in listdir(folder_path)
                          if isfile(join(folder_path, f)) and f.split(".png")[0].split("_")[-1] != 'None']
        self.country_labels_list = [x.split("_")[-1] for x in self.file_list]
        # one-hot-encode country labels
        self.ohe = OneHotEncoder(sparse=False)
        self.ohe.fit(np.array(self.country_labels_list).reshape(-1, 1))


    def __len__(self):
        return len(self.file_list)*32

    def nb_countries(self):
        return len(self.ohe.categories_[0])

    def __getitem__(self, idx):
        img_idx = idx // 32
        patch_idx = idx % 32

        f_name = self.file_list[img_idx]
        image_array = cv2.imread(join(self.folder_path, (f_name + ".png")))
        patches = patchify(image_array, (224, 224, 3), step=224)
        patches = patches.reshape((-1, 224, 224, 3))

        country_label_str = f_name.split("_")[-1]
        m = self.ohe.transform(np.array([country_label_str]).reshape(-1, 1))

        # return [patches, one_hot_encoded_country_label]
        return [patches[patch_idx, :, :, :], torch.Tensor(m)]
