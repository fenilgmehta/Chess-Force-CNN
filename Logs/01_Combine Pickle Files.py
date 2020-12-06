# Combine all Pickle File into ONE

# REFER: https://stackoverflow.com/questions/21887754/concatenate-two-numpy-arrays-vertically
import glob
import joblib
import numpy as np
from tqdm.autonotebook import tqdm
import time
import gc

all_files_list = glob.glob("../../Chess-Force-CNN-Dataset/04_pkl_data/*.pkl")
resa, resb = joblib.load(all_files_list[0])

cnt = 1
for i in tqdm(all_files_list[1:], ncols=100):
    cnt += 1
    atemp, btemp = joblib.load(i)
    print(str(cnt) + f" --- {atemp.shape} --- {btemp.shape}")
    resa = np.vstack((resa, atemp))  # this is 2-D
    resb = np.hstack((resb, btemp))  # this is 1-D
    del atemp, btemp
    gc.collect()  # Necessary as data is too large. So, RAM get consumed completely
    time.sleep(5)

joblib.dump((resa, resb), filename="complete_kingbase_dataset.pkl", compress=1)
