from os import listdir
from os.path import isfile, join
from pathlib import Path
import os


def go(folder_path, N):
    # N = nb of files per subfolder
    file_list = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    nb_of_folders = len(file_list) // N
    for i in range(nb_of_folders + 1):
        Path(folder_path + "/" + str(i)).mkdir(parents=True, exist_ok=True)

    c = 0
    for f in file_list:
        i = c // (N - 1)
        os.rename(join(folder_path, f), join(folder_path + "/" + str(i), f))
        c += 1


if __name__ == "__main__":
    go("D:/projets_perso/GeoG/datasets/v4", 2500)
