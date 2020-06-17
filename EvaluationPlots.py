import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pylab import *

class EvaluationPlots():
    def __init__(self, exact_pdh, sketch_pdh, base, num, ver, hor):
        self.exact_freq = exact_pdh['freq'].values
        self.exact_hhs = exact_pdh['val'].values
        self.topk = np.vstack((self.exact_hhs,self.exact_freq)).T
        self.estop = sketch_pdh.loc[:,['freq','val']].values
        self.k = self.topk.shape[0]
        self.esk = self.estop.shape[0]
        self.stream_1D = None
        self.base = base
        self.n = num
        self.l = None
        self.N = None
        self.ver = ver
        self.hor = hor

    def density(self, top, k):
        d = len(str(int(top[0][1])))
        m = int(top[0][1]) // 10**(d-1)
        i = 0
        x = []
        y = []
        while (i < k) & (m >= 0) & (d > 0):
            sum = 0
            idx = m * 10**(d-1)
            j = i
            while (i < k) & (top[i][1] >= idx):
                i += 1
                if i>=k:
                    break
            x.append(idx)
            y.append(i)
            #y.append(i-j)
            if m == 1:
                m = 9
                d -= 1
            else:
                m -=1
            if i>=k:
                break
        return x,y

    
    def plot_fig1(self, bar_width=0.05):
        x1,y1 = self.density(self.topk,self.k)
        x2,y2 = self.density(self.estop,self.esk)
#         mpl.rcParams["font.sans-serif"] = ["SimHei"]
#         mpl.rcParams["axes.unicode_minus"] = False
        bar_width1 = np.array(x1)*bar_width
        bar_width2 = np.array(x2)*bar_width
        plt.axes(xscale='log',yscale='log')
        plt.bar(x1, y1, bar_width1, align="center", color="c", label="Exact counting", alpha=0.5)
        plt.bar(x2+bar_width2, y2, bar_width2, color="b", align="center", label="Count sketch", alpha=0.5)
        plt.xlabel("Count")
        plt.ylabel("Number of cells")
        plt.legend()
        plt.show()
    
    def get_l_N(self):
        self.l = np.zeros((self.base**8,2))
        self.l[:,0] = self.base**8 - 1
        for i in range(self.esk):
            self.l[int(self.estop[i][0])][0] = i
            self.l[int(self.estop[i][0])][1] = self.estop[i][1]
        self.N = (self.ver)*(self.hor)*(self.n)/(self.base**8)
    
    def plot_fig2(self):
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for i in range(self.k):
            t = self.l[int(self.topk[i][0])][1]
            x1.append(i)
            y1.append(abs(self.topk[i][1]-t)/self.topk[i][1])
            if t != 0:
                x2.append(i)
                y2.append(abs(self.topk[i][1]-t)/self.topk[i][1])
        plt.figure()
        plt.plot(x1, y1, '.', color='c', alpha=0.5, label='Estimation error for all colors',markersize=1)
        plt.plot(x2, y2, '.', color='b', alpha=0.5, label='Estimation error for found colors',markersize=1)
        plt.legend(loc="upper right")
        plt.xlabel('Rank of frequency')
        plt.ylabel('Relative error')
        plt.ylim(bottom=-0.1, top = 1.2)
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
        plt.show()
    
    def plot_fig3(self):
        d1 = []
        y1 = []
        d2 = []
        y2 = []
        for i in range(self.k):
            t = self.l[int(self.topk[i][0])][1]
            y1.append(abs(self.topk[i][1]-t)/self.topk[i][1])
            d1.append((self.topk[i][1]-self.N)/self.N)
            if t != 0:
                y2.append(abs(self.topk[i][1]-t)/self.topk[i][1])
                d2.append((self.topk[i][1]-self.N)/self.N)
        plt.figure()
        plt.axes(xscale='log')
        plt.plot(d1, y1, '.', color='c', alpha=0.8, label='Estimation error for all colors',markersize=1)
        plt.plot(d2, y2, '.', color='b', alpha=0.8, label='Estimation error for found colors',markersize=1)
        plt.legend(loc="upper right")
        plt.xlabel('Delta of the color - log10')
        plt.ylabel('Relative error')
#        plt.xlim(0,)
        plt.show()
        
    def plot_fig4(self):
        d= len(str(self.k))-1
        m = self.k // 10**(d-1)
        idx0 = 0
        colors = ['b','g','r','c','m','y','k','w','burlywood','crimson','gray']
        plt.figure()
        for i in range(0,m):
            idx1 = min((i+1)*10**(d-1),self.k)
            num = sum(self.topk[idx0:idx1,1])
            x = []
            y = []
            for j in range(idx0,idx1):
                x.append(abs(self.l[int(self.topk[j][0])][0]-j))
                y.append(self.topk[j][1]/num)
            st = "ranks "+str(idx0)+"-"+str(idx1)
            plt.plot(x, y, '.', color=colors[i], alpha=0.8, label=st, markersize=1)
            idx0 = idx1
        plt.legend()
        plt.xlabel('Absolute error')
        plt.ylabel('Portion of colors')
        plt.ylim((0,0.05))
        plt.xlim(left=0,right=500)
        plt.show()
    
    def plot_all(self):
        self.plot_fig1()
        self.get_l_N()
        self.plot_fig2()
        self.plot_fig3()
        self.plot_fig4()
    
    