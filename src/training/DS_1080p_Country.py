from tqdm import tqdm
import torch
from os import listdir
from os.path import isfile, join
import cv2
from sklearn.preprocessing import OneHotEncoder
from patchify import patchify
import numpy as np
from torch.utils.data import Dataset
import pickle
from random import shuffle

from ..paths import ROOTDIR, DATADIR
from ..logging.utils import flog


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class PartitionedDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        nb_parts: int,
        size: tuple,
        offset=(0, 0),
        shuffle_buffer=False,
    ):
        self.size = size
        self.offset = offset
        self.patches_per_img = (1920 // size[0]) * (1080 // size[1])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.shuffle_buffer = shuffle_buffer
        self.nb_parts = nb_parts
        self.folder_path = folder_path

        with open(f"{ROOTDIR}training/country_tags.pkl", "rb") as f:
            country_tags = pickle.load(f)
        assert isinstance(country_tags, list)
        self.ohe = OneHotEncoder(sparse=False)
        self.ohe.fit(np.array(country_tags).reshape(-1, 1))

        self.file_list = [
            f.split(".png")[0]
            for f in listdir(folder_path)
            if isfile(join(folder_path, f))
            and f.split(".png")[0].split("_")[-1] in country_tags
        ]

        self.occurences = {key: 0 for key in country_tags}
        for f in self.file_list:
            tag = f.split(".png")[0].split("_")[-1]
            self.occurences[tag] += 1

        # print(self.occurences)

        # always shuffle file list globally at least once
        shuffle(self.file_list)

        to_drop = len(self.file_list) % nb_parts
        if to_drop != 0:
            self.file_list = self.file_list[:-to_drop]

        assert len(self.file_list) % nb_parts == 0

        self.nb_chunks = len(self.file_list) // self.nb_parts
        print(f"nombre d'imgs par chunk {self.nb_chunks}")
        self.chunks = list(chunks(self.file_list, self.nb_chunks))
        self.current_chunk_loaded = -1
        self.buffer = None

    def __len__(self):
        return len(self.file_list) * self.patches_per_img

    def nb_countries(self):
        return len(self.ohe.categories_[0])

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_img
        patch_idx = idx % self.patches_per_img
        requested_chunk = img_idx // len(self.chunks[0])
        requested_img_idx_in_chunk = img_idx % len(self.chunks[0])
        if requested_chunk != self.current_chunk_loaded:
            print(f"loading new chunk... {requested_chunk}")
            self.buffer = []

            for f_name in tqdm(self.chunks[requested_chunk]):
                label = f_name.split("_")[-1]
                m = self.ohe.transform(np.array([label]).reshape(-1, 1))

                try:
                    image_array = cv2.imread(join(self.folder_path, (f_name + ".png")))
                    image_array = image_array[self.offset[0] :, self.offset[1] :, :]
                except TypeError as e:
                    flog("!:!:!:!:!:!:!:!:!:")
                    flog(e)
                    flog(type(image_array))
                    flog(idx)
                    flog(self.folder_path)
                    flog(f_name)
                patches = patchify(image_array, self.size, step=self.size[0])
                patches = patches.reshape(
                    (-1, self.size[0], self.size[1], self.size[2])
                )

                for k in range(self.patches_per_img):
                    patch = patches[k, :, :, :]
                    self.buffer.append([torch.Tensor(patch), torch.Tensor(m)])

            if self.shuffle_buffer:
                shuffle(self.buffer)
            self.current_chunk_loaded = requested_chunk

        return self.buffer[
            requested_img_idx_in_chunk * self.patches_per_img + patch_idx
        ]


if __name__ == "__main__":
    from src.paths import ROOTDIR, DATADIR

    # train dataset
    train_ds = PartitionedDataset(
        f"{DATADIR}v3", 50, (456, 456, 3), offset=(65, 85), shuffle_buffer=True
    )
    print(len(train_ds))
