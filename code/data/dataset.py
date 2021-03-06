import os
import numpy as np

#os.environ['DATASET_PATH'] = r'../../../../../../bki/flatw/M21_1/'

class Dataset():
    def __init__(self,  DATASET_PATH, num, size = [1004,1344,35]):
        self.DATASET_PATH  = DATASET_PATH
        self.size = size
        self.ver, self.hor, self.layer = self.size
        self.data_path =None
        self.data = {}
        self.data1D = {}
        self.N_img = 0
# ===========================  INITIATING  ================================
        self.get_data_path(num)
# ===========================  FUNCTIONS  ================================

    def get_data_path(self, num, ini = 0):
        self.data_path = [os.path.join(self.DATASET_PATH, f) for f in os.listdir(self.DATASET_PATH) if f.endswith('.fw')]
        self.data_path = self.data_path[ini:ini + num]
        self.N_img = len(self.data_path)
        
    def get_data(self, idx):
        nptype=np.uint16
        with open(self.data_path[idx],'rb') as f_id:
            data = np.fromfile(f_id, count=np.prod(self.size),dtype = nptype)
            self.data[idx] = np.reshape(data, self.size)

    def get_test_data(self, idx, ver = 400, hor = 300):
        self.get_data(idx)
        self.data[idx] = self.data[idx][-ver:,-hor:,:]
        self.ver, self.hor = ver, hor