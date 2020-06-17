import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import *

class Evalsketch():
    def __init__(self, exact_pdh, cs_freq, cs_val, base, pca_comp):
        self.exact_freq = exact_pdh['freq'].values
        self.exact_val = exact_pdh['val'].values
        self.k = self.exact_freq.shape[0]
        self.cs_freq = cs_freq
        self.cs_val = cs_val
        # self.esk = self.cs_freq.shape[0]
        self.base = base
        self.pca_comp = pca_comp
        # self.pixel = pixel
        self.plot_log_hist()

    def plot_log_hist(self, bins = 20):
        plt.style.use('seaborn-deep')
        x = np.log10(self.exact_freq)
        y = np.log10(self.cs_freq)
        plt.hist([x, y], log = True, bins = 20,label=['Exact', 'Sketch'])
        plt.legend(loc='upper right')
        plt.xlabel('counts in log10')
        plt.ylabel('Number of Cells')
        plt.title('Log Histogram of Number of Cells for each count ')
    
    def get_relative_error(self):
        error = np.ones(self.k)
        for key, val in enumerate( self.exact_val ): 
            if val in self.cs_val:
                c0 = self.exact_freq[key]
                c1 = self.cs_freq[self.cs_val == val]
                # error[key] -= np.min([c1/c0,2])
                error[key] -= c1/c0
        return error
    
    def plot_rel_error(self):
        error = self.get_relative_error()
        plt.scatter(np.arange(self.k), error, s = 0.8)
        # plt.ylim(0,1)
        return error
