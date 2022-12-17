from random import randrange, shuffle, random
from tqdm import tqdm
import torch
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
from paths import ROOTDIR, DATADIR
import cv2


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class DS_Patches456_Partitioned:
    def __init__(self, folder_path: str, nb_chunks: int, stochastic: bool = False) -> None:
        self.folder_path = folder_path
        self.nb_chunks = nb_chunks
        self.stochastic = stochastic

        with open(f'{ROOTDIR}training/country_tags.pkl', 'rb') as f:
            country_tags = pickle.load(f)
        assert isinstance(country_tags, list)
        self.ohe = OneHotEncoder(sparse=False)
        self.ohe.fit(np.array(country_tags).reshape(-1, 1))

        self.file_list = sorted([f.split(".png")[0] for f in listdir(folder_path)
                          if isfile(join(folder_path, f))
                          and f.split(".png")[0].split("_")[-1] in country_tags])
        
        # always shuffle file list globally at least once
        # shuffle(self.file_list)

        self.occurences = {key: 0 for key in country_tags}
        for f in self.file_list:
            tag = f.split(".png")[0].split("_")[-1]
            self.occurences[tag] += 1

        print(self.occurences)

        self.nb_imgs_per_chunk = len(self.file_list) // self.nb_chunks
        print(f"nombre d'imgs par chunk {self.nb_imgs_per_chunk}")
        self.chunks = list(chunks(self.file_list, self.nb_imgs_per_chunk))
        self.current_chunk_loaded = -1
        self.buffer = None


    def _load_new_chunk(self, i):
        """
        Loads images of chunk number i in the buffer
        """
        print("loading new chunk..")
        self.buffer = []
        file_names = self.chunks[i]
        for f_name in file_names:
            img = cv2.imread(join(self.folder_path, (f_name + ".png")))
            lbl = f_name.split("_")[-1]
            self.buffer.append([img, lbl])

        self.current_chunk_loaded = i
        return

    def _get_item_in_buffer(self, item_idx):
        [img, lbl] = self.buffer[item_idx]
        if self.stochastic:
            # tag = self.ohe.inverse_transform(lbl.detach().cpu().numpy())
            proba = 50/self.occurences[lbl]
            # print(proba)
            if random() > proba:
                # get new random item_idx
                item_idx = randrange(start=0, stop=self.nb_imgs_per_chunk)
                return self._get_item_in_buffer(item_idx)
            else:
                return [img, lbl]
        else:
            return [img, lbl]


    def __getitem__(self, idx):
        chunk_idx = idx // self.nb_imgs_per_chunk #     99 // 5 = 19
        item_idx = idx % self.nb_imgs_per_chunk   #     99 % 5     = 4

        if chunk_idx != self.current_chunk_loaded:
            self._load_new_chunk(chunk_idx)

        assert self.current_chunk_loaded == chunk_idx
        
        return self._get_item_in_buffer(item_idx)
        


    def __len__(self):
        return len(self.file_list)*self.patches_per_img

    def nb_countries(self):
        return len(self.ohe.categories_[0])


if __name__ == "__main__" :
    d = DS_Patches456_Partitioned("/work/imvia/an3112sc/perso/GeoG/datasets/v3_eval_patches", 2, stochastic=True)
    
    
    for i in range(600, 620):
        img, lbl = d[i]
        print(lbl)