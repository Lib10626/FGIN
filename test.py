import scipy.io as sio
import numpy as np


pfc_feat_path_train = 'data/CUB2011/pfc_feat_train.mat'
pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)[:,:512]
print(pfc_feat_data_train.shape)