# -*- coding: utf-8 -*-
import sys
import os
import math
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import umap.umap_ as uma
from collections import Counter
sys.path.insert(0, r'C:\Users\viska\Documents\AceCan')
os.chdir(r"C:\Users\viska\Documents\AceCan")
data_dir = r'.\bki'
output_path= os.getcwd()
from timeit import default_timer as timer
from Dataset import dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn import preprocessing

from csh.csvec.csvec import CSVec 
from CountSketch import CountSketch

#################### Stream 1D ########################

def execute_prepro(data_dir, base, num, pca_comp, topk, test, percentage = 0.01):
    data1Ds, pc = process_dataset(data_dir, base, num, pca_comp, topk, test)
    stream_1D, mask = process_pca(data1Ds, pc, num, base, pca_comp)
    HH = process_count0(stream_1D, topk)
    exact_pdh = process_HH(HH, base, pca_comp, percentage)
    if len(exact_pdh) > 2000:
        print('len(exact_pdh) > 2000, umap is skipped')
        return exact_pdh, stream_1D, mask
    process_umap(exact_pdh, pca_comp)
    return exact_pdh, stream_1D, mask


############## Process Data #################
def process_dataset(data_dir, base, num, pca_comp, topk, test):   
    images = dataset(data_dir, num)
    num  = images.N_img
#     pca_combined = np.zeros([images.layer, images.layer])
    print(f'Processing {num} [test :{test}] images with original size {images.size} ')
    with ThreadPoolExecutor() as executor: 
        futures = []
        for idx in range(num):
            futures.append(executor.submit(lambda x: run_step_multiple(x, images, test), idx))
            # print(f" No.{idx} image is loaded")
        mul_comb = np.zeros([images.layer,images.layer])
        for future in as_completed(futures):
            mul = future.result()
            mul_comb += mul
        pc = run_step_pc(mul_comb, pca_comp)
    return images.data1D, pc

def run_step_multiple(idx, images, test = 'False'):
    if test:
        images.get_test_data(idx)
    else:
        images.get_data(idx)
    images.data1D[idx] = np.reshape(images.data[idx], [images.ver*images.hor, images.layer]).astype(float)
    return images.data1D[idx].T.dot(images.data1D[idx])

def run_step_pc(mul_comb, pca_comp):
    # use svd since its commputational faster
    print("=============== run step PCA ===============")
    u,s,v = np.linalg.svd(mul_comb)
    assert np.allclose(u, v.T)
    print('Explained Variance Ratio', np.round(s/sum(s),3))
    pc = u[:,:pca_comp]
    return pc

#################### PCA, Intensity, Norm ########################
def process_pca0(data1Ds, pc, num, base, pca_comp):
    with ThreadPoolExecutor() as executor: 
        futures = []
        for idx in range(num):
            futures.append(executor.submit(lambda x: run_step_pc_transform(x, data1Ds, pc), idx))
            # print(f" No.{idx} image is transformed")
        pca_results = np.zeros([1,pca_comp])
        for future in as_completed(futures):
            pca_result = future.result()
            pca_results = np.vstack((pca_results,pca_result))
    pca_results = pca_results[1:,:]
    print('========= Intensity ==============')
    intensity = (np.sum(pca_results**2, axis = 1))**0.5
    # cutoffH = np.mean(intensity).round()
    cutoffH = np.exp(5.8).round()
    print('cutoffH is set to be mean', cutoffH)
    stream_1D, mask = run_step_norm(pca_results, intensity, cutoffH, base, pca_comp)
    return stream_1D, mask

def process_pca(data1Ds, pc, num, base, pca_comp):
    pca_results = run_step_pc_transform(0, data1Ds, pc)
    for i in range(1,num):
        pca_result = run_step_pc_transform(i, data1Ds, pc)
        pca_results = np.vstack((pca_results,pca_result))
    # pca_results = pca_results[1:,:]
    print('========= Intensity ==============')
    intensity = (np.sum(pca_results**2, axis = 1))**0.5
    # cutoffH = np.mean(intensity).round()
    cutoffH = run_step_cutoff(intensity)
    print('cutoffH is set to be mean', cutoffH)
    stream_1D, mask = run_step_norm(pca_results, intensity, cutoffH, base, pca_comp)
    return stream_1D, mask

def run_step_pc_transform(x, data1Ds, pc):
    return data1Ds[x].dot(pc)

def run_step_cutoff(intensity, N_bins = 100, N_sigma = 3 ):
    para = np.log(intensity[intensity > 0])
    (x,y) = np.histogram(para, bins = N_bins)
    y = (y[1]-y[0])/2 + y[:-1]
    assert len(x) == len(y)
    x_max =  np.max(x)
    x_half = x_max//2
    mu = y[x == x_max]
    sigma = abs(y[abs(x - x_half).argmin()] -mu)
    cutoff_log = N_sigma* sigma + mu
    cutoff = int(np.exp(cutoff_log).round())
    return cutoff

def run_step_norm(pca_result, intensity, cutoffH, base, pca_comp):
    mask = intensity > cutoffH
    # print('norm length',np.sum(mask))
    norm_data = np.divide(pca_result[mask], intensity[mask][:,None])
    print('norm_data', np.min(norm_data), np.max(norm_data))
    pca_rebin = np.trunc((norm_data + 1) * base/2)
    print('rebin, min/mac', np.min(pca_rebin), np.max(pca_rebin))
    stream_1D = 0
    for comp in range(pca_comp):
        stream_1D = stream_1D + pca_rebin.T[comp]*base**comp
    return stream_1D, mask

#################### 1D Stream #####################
def inverse_mapcode(stream_1D, base, pca_comp):
    stream_1D = np.array(stream_1D)
    inverted_pca = np.zeros((pca_comp,len(stream_1D)))
    for i in range(pca_comp):
        inverted_pca[i] = stream_1D % base
        stream_1D = stream_1D // base
    return inverted_pca

def process_stream_1D(stream_1D, base, pca_comp, topk = 10000, percentage = 0.01 ):
    c = Counter(stream_1D)
    HH = c.most_common(topk)
    exact_a = np.array(HH)
    exact_val, exact_freq = exact_a[:,0].astype('Int64'), exact_a[:,1].astype('Int64')
    exact_pca = inverse_mapcode(exact_val, base,  pca_comp)
    exact_pd = pd.DataFrame(exact_pca.T, columns = range(pca_comp))
    exact_pd['freq'] = np.abs(exact_freq)
    exact_pd['val'] = exact_val
    high_cut = exact_pd['freq'][0]* percentage
    print(high_cut)
    exact_pdh = exact_pd[exact_pd['freq']> high_cut]
    print('#exact_pdh', len(exact_pdh), high_cut)
    print(exact_pd)
    # np.savetxt(f'{name}/exact_pdh_DO.cvs', exact_pdh)
    return exact_pdh


#################### UMAP ###########################
def process_umap(exact_pdh, pca_comp, scale = 500):
    umapH = uma.UMAP()
    umap_result = umapH.fit_transform(exact_pdh[list(range(pca_comp))])
    freqlist  = exact_pdh['freq']
    lw = (freqlist/freqlist[0])**2
    u1 = umap_result[:,0]
    exact_pdh['u1'] = u1
    u2 = umap_result[:,1]
    exact_pdh['u2'] = u2
    plt.scatter(u1, u2, s = scale*lw)
    return None

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300) 
# tsne_results = tsne.fit_transform(exact_pdh[range(8)])
# exact_pdh['tsne1'] = tsne_results[:,0]
# exact_pdh['tsne2'] = tsne_results[:,1]
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne1", y="tsne2",
# #     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=exact_pdh,
#     legend="full",
#     alpha=0.3
# )
# # np.savetxt(f'{name}/exact_pdh', exact_pdh)
###################### Exact Counting ###########################
def process_count0(stream_1D, topk = 10000):
    start =  timer()
    c = Counter(stream_1D)
    HH = np.array(c.most_common(topk))
    end =  timer()
    print('Exact Counting time', np.round((end - start),2))
    return HH


#################### CS ###########################
def process_countsketch(d, vec, base, topk, col_range, row_range, display, name):
    print('################################## Running Count Sketch###################################')
    #start count sketch  
    sketch_times  = []
    for row in row_range:     
        for col in col_range:
            sketch_hhs, sketch_freqs, sketch_time = run_step_sketch(vec, d, topk, col,row)     
            print(f'################################## Count Sketch row {row} x col {col} topk {topk} Results###################################')
            print (f'sketch_hhs_r{row}c{col}b{base}k{topk}',sketch_hhs[:display])
            print (f'sketch_freqs_r{row}c{col}b{base}k{topk}', np.absolute(sketch_freqs[:display]))
#             with open(f'{name}\\sketch_hhs_r{row}c{col}k{topk}.txt', 'w') as file:
#                 file.write(json.dumps([sketch_hhs.tolist(),sketch_freqs.tolist()]))
#             sketch_times.append( [row, col, round(sketch_time,3)])
#     [sketch_times[k][2] for k in range(len(sketch_times))]
    np.savetxt(f'{name}\\sketch_hhs_r{row}c{col}k{topk}.txt', [sketch_hhs,sketch_freqs])
#     np.savetxt(f'{name}\\sketch_times_b{base}k{topk}.txt', sketch_times)
    return sketch_times


        
def run_step_sketch(vec, d, topk, col, row):
    cs = CSVec(d = d , c=col, r=row)
    start_sketch = timer()
    count_sketch = CountSketch(vec, cs, topk)
    count_sketch.run()
    print(f'HHs processed')
    end_sketch = timer()
    sketch_time = end_sketch - start_sketch   
    sketch_hhs, sketch_freqs = count_sketch.get_hhs()
    return sketch_hhs, sketch_freqs, sketch_time


#################### Heavy Hitters ########################

def process_HH(HH, base, pca_comp, percentage = 0.01 ):
    exact_val, exact_freq = HH[:,0].astype('Int64'), HH[:,1].astype('Int64')
    exact_pca = inverse_mapcode(exact_val, base,  pca_comp)
    exact_pd = pd.DataFrame(exact_pca.T, columns = range(pca_comp))
    exact_pd['freq'] = np.abs(exact_freq)
    exact_pd['val'] = exact_val
    high_cut = exact_pd['freq'][0]* percentage
    print(high_cut)
    exact_pdh = exact_pd[exact_pd['freq']> high_cut]
    print('#exact_pdh', len(exact_pdh), high_cut)
    print(exact_pd)
    # np.savetxt(f'{name}/exact_pd.cvs', exact_pd)
    return exact_pdh

def process_sketch_HH(HH, base, pca_comp ):
    exact_val, exact_freq = HH[0].cpu().numpy() , HH[1].cpu().numpy()
    exact_pca = inverse_mapcode(exact_val, base,  pca_comp)
    exact_pd = pd.DataFrame(exact_pca.T, columns = range(pca_comp))
    exact_pd['freq'] = np.abs(exact_freq)
    exact_pd['val'] = exact_val
    print(exact_pd)
    # np.savetxt(f'{name}/exact_pdh_DO.cvs', exact_pdh)
    return exact_pd


#################### Intensity ########################

def process_intensity(pca_result, intensity, cutoffH, base, pca_comp):
    mask = intensity > cutoffH
    print('norm length',np.sum(mask))
    norm_data = np.divide(pca_result[mask], intensity[mask][:,None])
    print('norm_data', np.min(norm_data), np.max(norm_data))
    pca_rebin = np.trunc((norm_data + 1) * base/2)
    print('rebin, min/mac', np.min(pca_rebin), np.max(pca_rebin))
    mapcode = 0
    for comp in range(pca_comp):
        mapcode = mapcode + pca_rebin.T[comp]*base**comp
    # print(np.min(mapcode))
    #      print(mapcode, self.stream_1Ds)
    stream_1D = mapcode
    # print(np.min(stream_1D))
    return stream_1D, norm_data, mask


def get_intensity_hist(intensity, cutoff_log = 6, range = (5,10)):
    para = np.log(intensity+ 0.01)
    histdata = plt.hist(para, bins = 100, range = range, color = 'gold')
    median_val = np.median(intensity).round()
    mean_val = np.mean(intensity).round()
    cutoff = np.round(np.e**cutoff_log)
    plt.axvline(np.log(mean_val), color = 'g', label = f'mean = {mean_val}')
    plt.axvline(np.log(median_val), color = 'b', label = f'median = {median_val}')
    plt.axvline(cutoff_log, color = 'r', label = f'cutoff = {cutoff}' )
    plt.title('Histogram of Intensity Cutoff ')
    plt.xlabel('log(intensity)')
    plt.legend()
    return cutoff, mean_val, median_val



#################### Testing #######################

def test_mul_comb(mul_comb, pc, pca_comp):
    u,s,v = np.linalg.svd(mul_comb)
    assert np.allclose(u[:,:pca_comp], pc)
    plt.plot(np.log(s))
    plt.ylabel('log(eigenvalues)')
    plt.xlabel('layers')

def test_rebin(self, norm_data, mask, base, idx = 0, bg = -0.1):
    masked_rebin = np.ones([pixel * num, pca_comp])* (bg)
    pca_rebin = np.trunc((norm_data + 1) * base/2)
    print('rebin, min/mac', np.min(pca_rebin), np.max(pca_rebin))
    masked_rebin[mask] = pca_rebin
    plot_rebin_data(self, masked_rebin, idx, bg, base)
    return masked_rebin

#################### K-Means ###########################
from sklearn.cluster import KMeans
import seaborn as sns

def process_kmean(exact_pdh,  N_clusters = [3,10], u1 = 'u1', u2 = 'u2', k_cluster = 'kmean', weighted = False):
    if weighted:
        weight = exact_pdh['freq'].values
    else:
        weight = None
    k_names = []
    umap_result = exact_pdh.loc[:,[u1, u2 ]].values
    l_cluster = len(N_clusters)
    print(l_cluster)
    f, axes = plt.subplots(1,l_cluster,figsize= (16,5) )
    for i in range(l_cluster):
        N_cluster = N_clusters[i]
        k_name = f'k{N_cluster}'
        k_names += [k_name]
        kmeans = KMeans(n_clusters=N_cluster)
        kmeans.fit(umap_result, sample_weight = weight)
        y_km = kmeans.predict(umap_result)
        y_km += 1
        exact_pdh[k_name] = y_km 
        sns.scatterplot(
            x=u1, y=u2,
            hue= k_name ,
            palette=sns.color_palette("muted", N_cluster),
            data=exact_pdh,
            legend="full",
            ax = axes[i]
            # alpha=0.3 
            )
    # print(exact_pdh.loc[[0,1]])
    return 

def plot_kmean_clusters(stream_1D, mask, exact_pdhh,  k_cluster, val = 'val', bg = -1, color = 0, sgn = -1 ):
    HHvals = np.array(exact_pdhh[val])
    HHcluster = np.array(exact_pdhh[k_cluster])
    masked_streams = np.zeros(np.shape(stream_1D))
    for idx, val in enumerate( HHvals): 
        label = HHcluster[idx]
        masked_streams = np.where(stream_1D != val, masked_streams, color -sgn * label)
    final_umap = np.ones(mask.shape) * bg
    final_umap[mask] = masked_streams
    return final_umap, masked_streams 


def plot_kmean_binary(stream_1D, mask, HHvals, val = 'val',  bg = -1, color = 1 ):
    masked_streams = np.zeros(np.shape(stream_1D))
    for val in HHvals: 
        masked_streams = np.where(stream_1D != val, masked_streams, color)
    final_umap = np.ones(mask.shape) * bg
    # binary_umap = np.ones(mask.shape) * bg
    # masked_binary = (masked_streams > 0) * 1
    final_umap[mask] = masked_streams
    # binary_umap[mask] = masked_binary
    return final_umap, masked_streams 


############################## Final Filtered Results #####################################
def plot_HH_masked(stream_1D, mask, cluster_dict, bg, c_ini = 5, c_fin = 20):
    masked_streams = np.zeros(np.shape(stream_1D))    
    print('ini,fin', c_ini, c_fin )
    for i in range(c_ini, c_fin ):
        masked_stream_1D = np.zeros(np.shape(stream_1D)) 
        vals = np.array(cluster_dict[f'{i + 1}'])
        for val in vals:
            masked_stream_1D = np.where(stream_1D != val, masked_stream_1D, c_fin - i)
        masked_streams += masked_stream_1D
    print(np.min(masked_streams),np.max(masked_streams))
    final_results = np.ones(mask.shape) * bg
    binary_results = np.ones(mask.shape) * bg
    masked_binary = (masked_streams > 0) * 1
    print(np.max(masked_binary))
    final_results[mask] = masked_streams
    binary_results[mask] = masked_binary
    return final_results, binary_results

############################## Glueviz Functions #########################################
def get_cluster_idx(data0, sub_range, exact_pdh, col = 'col13'):
    cluster = {}
    i = 1
    for subset in sub_range:
        layer_data = data0.subsets[subset]
        cluster[i] = layer_data[col].astype(int)
        i+=1
    exact_pdh['cluster'] = np.zeros(len(exact_pdh)).astype(int)
    for key in range(1,max(sub_range)+2):
        exact_pdh.loc[cluster[key],'cluster'] = int(key)
    return None

def plot_umap_clusters(stream_1D, exact_pdhh, bg = -1 ):
    HHvals = np.array(exact_pdhh['val'])
    HHcluster = np.array(exact_pdhh['cluster'])
    masked_streams = np.zeros(np.shape(stream_1D))
    for idx, val in enumerate( HHvals): 
        label = HHcluster[idx]
        masked_streams = np.where(stream_1D != val, masked_streams, label)
    final_umap = np.ones(mask.shape) * bg
    binary_umap = np.ones(mask.shape) * bg
    masked_binary = (masked_streams > 0) * 1
    final_umap[mask] = masked_streams
    binary_umap[mask] = masked_binary
    return final_umap, binary_umap,  masked_streams 




# def process_eigen()

def main():   
    sys.path.insert(0, r'C:\Users\viska\Documents\AceCan')
    os.chdir(r"C:\Users\viska\Documents\AceCan")
    Output_path = r"C:\Users\viska\Documents\AceCan\UMAP"
    try:
        os.mkdir(Output_path)
    except:
        None
    load_data = 1
    num = 40
    base = 8
    topk = 10000
    col_range =  [8000]
    row_range = [5]
    display = 20
    pca_comp = 8
    d = base ** pca_comp 
    data_dir = r'.\bki'    
    test = 0
    name = os.path.join(Output_path, f'b{base}topk{topk}N{num}test')
    if load_data == True:
        try:
            os.mkdir(name)
        except:
            print('overwriting directory')
        print(f'Output directory: {name}')
        print('################################## Running PCA Preprocessing ###################################')
#          exact_HH, vec, 
        stream_concat = process_HH(data_dir, base, num, pca_comp, topk, test)
#         np.savetxt(f'{name}\\exact.txt', exact_HH)
#         np.savetxt(f'{name}\\vec_b{base}.txt',vec)   
#         np.savetxt(f'{name}\\stream_concat{base}.txt',stream_concat) 
#         sketch_times = process_countsketch(d, vec, base, topk, col_range, row_range, display, name)
#         print(sketch_times)
    elif load_data == False:
        # load preprocessed data
        print('################################## Loading 1D frequency vector ###################################')
#         with open(f'{name}\\vecs_b{base}.txt') as json_file:
#             vec = json.load(json_file)  
        vec = np.loadtxt(f'{name}\\vec_b{base}.txt')
        print('vec',vec)
        sketch_times = process_countsketch(d, vec, base, topk, col_range, row_range, display, name)
        print(sketch_times)
#         print('################################## Analyzing Runtime ###################################')      
#        with open(f'{name}\\exact_time_b{base}k{topk}.txt') as json_file:
#            exact_time = json.load(json_file)
#        finish(exact_time, sketch_times)
    else:
        raise ValueError('load_data boolean value not set')
#    finish(exact_time, sketch_times)
#    plots = Plotting()
#    plots.layer_plot_4()
    
    
if __name__ == "__main__":
    main()
    
    

#def run_step_sketch(converts, d, topk, col, row):
#    cs = CSVec(d = d , c=col, r=row)
#    start_sketch = timer()
#    for idx in range(len(converts)):
#        count_sketch = CountSketch(converts[idx].vec, cs, topk = topk)
#        count_sketch.run()
#        print(f'No.{idx} hhs processed')
#    end_sketch = timer()
#    sketch_time = end_sketch - start_sketch   
#    sketch_hhs, sketch_freqs = count_sketch.get_hhs()  
#    return sketch_hhs, sketch_freqs, sketch_time



# def run_step_sketch(vecs, d, topk, col, row):
#     cs = CSVec(d = d , c=col, r=row)
#     start_sketch = timer()
#     for key, vec in vecs.items():
#         count_sketch = CountSketch(vec, cs, topk = topk)
#         count_sketch.run()
#         print(f'No.{key} hhs processed')
#     end_sketch = timer()
#     sketch_time = end_sketch - start_sketch  
#     sketch_hhs, sketch_freqs = count_sketch.get_hhs()  
#     return sketch_hhs, sketch_freqs, sketch_time


# def process_vecs(converts):
#     vecs = {}
#     for idx in range(len(converts)):
#         vecs[idx] = converts[idx].vec.tolist()
#     return vecs

# def process_countsketch(d, vecs, base, topk, col_range, row_range, display, name):
# #    print('##################################Running Count Sketch###################################')
#     #start count sketch  
#     sketch_times  = []
#     for row in row_range:     
#         for col in col_range:
#             sketch_hhs, sketch_freqs, sketch_time = run_step_sketch(vecs = vecs, d = d, topk = topk, col = col, row = row)       
#             print(f'################################## Count Sketch row {row} x col {col} topk {topk} Results###################################')
# #            print (f'sketch_hhs_r{row}c{col}b{base}k{topk}',sketch_hhs[:display])
# #            print (f'sketch_freqs_r{row}c{col}b{base}k{topk}', np.absolute(sketch_freqs[:display]))
#             with open(f'{name}\\sketch_hhs_r{row}c{col}k{topk}.txt', 'w') as file:
#                 file.write(json.dumps([sketch_hhs.tolist(),sketch_freqs.tolist()]))
#             sketch_times.append( [row, col, round(sketch_time,3)])
# #     [sketch_times[k][2] for k in range(len(sketch_times))]
#     np.savetxt(f'{name}\\sketch_times_b{base}k{topk}.txt', sketch_times)
#     return sketch_times

# from __future__ import print_function
# import time
# import numpy as np
# import pandas as pd
# # from sklearn.datasets import fetch_mldata
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# %matplotlib inline
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# from matplotlib.colors import LogNorm
# import umap.umap_ as uma
# import math
# from collections import Counter