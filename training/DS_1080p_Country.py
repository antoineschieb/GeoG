from tqdm import tqdm
import torch
from os import listdir
from os.path import isfile, join
import cv2
from sklearn.preprocessing import OneHotEncoder
from patchify import patchify
import numpy as np
from torch.utils.data import Dataset
import time
from random import shuffle



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class PartitionedDataset(Dataset):
    def __init__(self, folder_path, nb_parts, shuffle_buffer=False, ohe=None):
        print("init")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.shuffle_buffer = shuffle_buffer
        self.nb_parts = nb_parts
        self.folder_path = folder_path

        self.file_list = [f.split(".png")[0] for f in listdir(folder_path)
                          if isfile(join(folder_path, f)) and f.split(".png")[0].split("_")[-1] != 'None']
        if shuffle_buffer:
            shuffle(self.file_list)

        to_drop = len(self.file_list) % nb_parts
        if to_drop != 0:
            self.file_list = self.file_list[:-to_drop]

        assert len(self.file_list) % nb_parts == 0

        self.nb_chunks = len(self.file_list) // self.nb_parts
        self.chunks = list(chunks(self.file_list, self.nb_chunks))
        self.current_chunk_loaded = -1

        self.buffer = None

        self.country_labels_list = [x.split("_")[-1] for x in self.file_list]
        # one-hot-encode country labels
        if ohe is not None:
            self.ohe = ohe
        else:
            self.ohe = OneHotEncoder(sparse=False)
            self.ohe.fit(np.array(self.country_labels_list).reshape(-1, 1))

    def __len__(self):
        return len(self.file_list)*32

    def nb_countries(self):
        return len(self.ohe.categories_[0])

    def __getitem__(self, idx):
        img_idx = idx // 32
        patch_idx = idx % 32
        requested_chunk = img_idx // len(self.chunks[0])
        requested_img_idx_in_chunk = img_idx % len(self.chunks[0])
        if requested_chunk != self.current_chunk_loaded:
            print(f"loading new chunk... {requested_chunk}")
            self.buffer = []

            for f_name in self.chunks[requested_chunk]:
                label = f_name.split("_")[-1]
                m = self.ohe.transform(np.array([label]).reshape(-1, 1))

                image_array = cv2.imread(join(self.folder_path, (f_name + ".png")))
                patches = patchify(image_array, (224, 224, 3), step=224)
                patches = patches.reshape((-1, 224, 224, 3))

                for k in range(32):
                    patch = patches[k, :, :, :]
                    self.buffer.append([patch, torch.Tensor(m)])

            if self.shuffle_buffer:
                shuffle(self.buffer)
            self.current_chunk_loaded = requested_chunk


        return self.buffer[requested_img_idx_in_chunk * 32 + patch_idx]





# ========================================================================================


class DS_1080p_Country(Dataset):
    def __init__(self, folder_path, ohe=None):
        self.folder_path = folder_path
        self.file_list = [f.split(".png")[0] for f in listdir(folder_path)
                          if isfile(join(folder_path, f)) and f.split(".png")[0].split("_")[-1] != 'None']

        self.image_arrays = []
        for f_name in tqdm(self.file_list, total=len(self.file_list)):
            image_array = cv2.imread(join(self.folder_path, (f_name + ".png")))
            self.image_arrays.append(image_array)

        self.country_labels_list = [x.split("_")[-1] for x in self.file_list]
        # one-hot-encode country labels
        if ohe is not None:
            self.ohe = ohe
        else:
            self.ohe = OneHotEncoder(sparse=False)
            self.ohe.fit(np.array(self.country_labels_list).reshape(-1, 1))

    def __len__(self):
        return len(self.file_list)*32

    def nb_countries(self):
        return len(self.ohe.categories_[0])

    def __getitem__(self, idx):
        img_idx = idx // 32
        patch_idx = idx % 32

        image_array = self.image_arrays[img_idx]
        patches = patchify(image_array, (224, 224, 3), step=224)
        patches = patches.reshape((-1, 224, 224, 3))

        f_name = self.file_list[img_idx]
        country_label_str = f_name.split("_")[-1]
        m = self.ohe.transform(np.array([country_label_str]).reshape(-1, 1))

        return [patches[patch_idx, :, :, :], torch.Tensor(m)]
