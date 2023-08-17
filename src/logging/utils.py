import os
import glob
from pathlib import Path as P

from ..paths import ROOTDIR, DATADIR


def flog(strr, run_dir=None):
    if run_dir is None:
        # Get list of all files only in the given directory
        list_of_files = filter(
            os.path.isdir, glob.glob(str(P.joinpath(P(ROOTDIR), "runs/*")))
        )
        # Sort list of files based on last modification time in ascending order
        list_of_files = sorted(list_of_files, key=os.path.getmtime, reverse=True)

        run_dir = list_of_files[0]  # newest

    d = P.joinpath(P(ROOTDIR), "runs", run_dir)

    strr = str(strr)
    with open(P.joinpath(d, "flogs.txt"), "a") as myfile:
        myfile.write(strr + "\n")
    print(f"flogs: {strr}")
    return
