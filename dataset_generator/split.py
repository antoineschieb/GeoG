from random import random
import os
import shutil
from pathlib import Path as P
from tqdm import tqdm


ds_path = P("/work/imvia/an3112sc/perso/GeoG/datasets/1080/v3")
dsp = str(ds_path)
p = ds_path.glob("**/*")
files = [x for x in p if x.is_file()]
for full_file_path in tqdm(files):
    f = full_file_path.name
    full_file_path = str(full_file_path)
    # print(full_file_path)
    if random() < 0.8:
        shutil.move(
            full_file_path, "/work/imvia/an3112sc/perso/GeoG/datasets/1080/train/" + f
        )
    else:
        shutil.move(
            full_file_path, "/work/imvia/an3112sc/perso/GeoG/datasets/1080/test/" + f
        )
