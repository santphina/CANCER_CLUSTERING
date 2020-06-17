# -*- coding: utf-8 -*-
import sys
import os
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
    return images.data1D, pc, mul_comb

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

#################### PCA ########################
def process_pca(data1Ds, pc, num, pca_comp):
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
    return intensity, pca_results

def run_step_pc_transform(x, data1Ds, pc):
    return data1Ds[x].dot(pc)

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
    # np.savetxt(f'{name}/exact_pdh', exact_pdh)
    return exact_pdh

#################### UMAP ###########################
def process_umap(exact_pdh, pca_comp, scale = 500):
    umapH = uma.UMAP()
    umap_result = umapH.fit_transform(exact_pdh[list(range(pca_comp))])
    freqlist  = exact_pdh['freq']
    lw = (freqlist/freqlist[0])**2
    plt.scatter(umap_result[:,0], umap_result[:,1], s = scale*lw)
    return umap_result

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
        stream_concat = process_dataset(data_dir, base, num, pca_comp, topk, test)
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
