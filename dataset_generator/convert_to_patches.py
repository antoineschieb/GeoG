from tqdm import tqdm
import os
from pathlib import Path as P
from patchify import patchify
import cv2 as cv

if __name__ == "__main__":
    size = (456, 456, 3)
    ds_path = P("/work/imvia/an3112sc/perso/GeoG/datasets/1080/train")
    p = ds_path.glob("**/*")
    files = [x for x in p if x.is_file()]
    for full_file_path in tqdm(files):
        f = full_file_path.stem
        full_file_path = str(full_file_path)

        suffix = f.split("_")[-1]
        prefix = f[:-3]

        img = cv.imread(full_file_path)
        patches = patchify(img, size, size[0])
        patches = patches.reshape((-1, size[0], size[1], size[2]))

        for i in range(8):
            patch = patches[i, :, :, :]
            new_name = prefix + "_" + str(i) + "_" + suffix
            loc = ds_path.parent.parent.joinpath("456/train").joinpath(
                new_name + ".png"
            )
            cv.imwrite(str(loc), patch)
