# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import json
sys.path.insert(0, r'C:\Users\viska\Documents\AceCan')
os.chdir(r"C:\Users\viska\Documents\AceCan")
data_dir = r'.\bki'
output_path= os.getcwd()
from timeit import default_timer as timer
from Dataset import dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn import preprocessing
from ExactCount import ExactCount
from csh.csvec.csvec import CSVec 
from CountSketch import CountSketch
from collections import Counter
# from Convert import Convert

#from plotting import Plotting

def process_dataset(data_dir, base, num, pca_comp, topk, test):   
    images = dataset(data_dir, num)
    pca_combined = np.zeros([images.layer, images.layer])
    print(f'ThreadPoolExecutor Initiated for {images.N_img} images with size {images.size} ')
    with ThreadPoolExecutor() as executor: 
        futures = []
        for idx in images.idxs:
            futures.append(executor.submit(lambda x: run_step_covariance(x, images, test), idx))
            print(f" No.{idx} image is loaded")
        cov_combined = np.zeros([images.layer,images.layer])
        result = []
        for future in as_completed(futures):
            result = future.result()
            cov_combined += result
#             print('processed result',result[:5,:5])
        print("Done?", future.done())
        pc_eigens, pc_vals = images.get_pc_eigens(cov_combined, pca_comp)
#         for val images.data.items()
#         print('pc_eigens',pc_eigens[0:2,:], np.shape(pc_eigens))
        futures = []
        for idx in images.idxs:
            futures.append(executor.submit(lambda x: run_step_pca(x, pc_eigens, images, base, pca_comp), idx))
        '''
        stream_concat = []
        stream_1D = []
        vec = np.zeros(base**pca_comp)    
        print('################################## Preparing 1D frequency vector ###################################')
        '''
        for future in as_completed(futures):
            stream_1D = future.result()
            print(stream_1D)
        '''
            stream_concat = np.append(stream_concat,stream_1D)
            for item in stream_1D:
                vec[item] +=1
        print("Done?", future.done())
        print('non-zero in vec', np.sum(vec!=0))            
#             print('processed result',result[[0,1,-2,-1]])        
    print(np.shape(stream_concat))
    print(stream_concat)
    exact_HH = run_exact_count(stream_concat, topk)
    print('################################## Exact Count Done ###################################')
    print('exact_HH',exact_HH[:10])
    return exact_HH, vec, stream_concat
    '''
    return None

def process_dataset_test(data_dir, base, num, pca_comp, topk, test):   
    images = dataset(data_dir, num)
    pca_combined = np.zeros([images.layer, images.layer])
    print(f'ThreadPoolExecutor Initiated for {images.N_img} images with size {images.size} ')
    with ThreadPoolExecutor() as executor: 
        futures = []
        for idx in images.idxs:
            futures.append(executor.submit(lambda x: run_step_covariance(x, images, test), idx))
            print(f" No.{idx} image is loaded")
        cov_combined = np.zeros([images.layer,images.layer])
        result = []
        for future in as_completed(futures):
            result = future.result()
            cov_combined += result
#             print('processed result',result[:5,:5])
        print("Done?", future.done())
        pc_eigens, pc_vals = images.get_pc_eigens(cov_combined, pca_comp)
        print(np.log(pc_vals))
#         for val images.data.items()
#         print('pc_eigens',pc_eigens[0:2,:], np.shape(pc_eigens))
        '''
        futures = []
        for idx in images.idxs:
            futures.append(executor.submit(lambda x: run_step_pca(x, pc_eigens, images, base, pca_comp), idx))
 
        stream_concat = []
        stream_1D = []
        vec = np.zeros(base**pca_comp)    
        print('################################## Preparing 1D frequency vector ###################################')
        
        for future in as_completed(futures):
            stream_1D = future.result()
            print(stream_1D)
  
            stream_concat = np.append(stream_concat,stream_1D)
            for item in stream_1D:
                vec[item] +=1
        print("Done?", future.done())
        print('non-zero in vec', np.sum(vec!=0))            
#             print('processed result',result[[0,1,-2,-1]])        
    print(np.shape(stream_concat))
    print(stream_concat)
    exact_HH = run_exact_count(stream_concat, topk)
    print('################################## Exact Count Done ###################################')
    print('exact_HH',exact_HH[:10])
    return exact_HH, vec, stream_concat
    '''
    return None

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

def run_step_covariance(idx, images, test = 'False'):
    if test:
        images.get_test_data(idx)
    else:
        images.get_data(idx)
    cov_matrix = images.get_cov_matrix(idx)
#     print('====================',np.round(cov_matrix[:5,:5]))
    return cov_matrix


def run_exact_count(stream_concat, topk):
    c = Counter(stream_concat)
    print(f'=================================== Exact Counting {topk} ================================')
    return c.most_common(topk)

  
    
def run_step_pca(idx, pc_eigens, images, base, pca_comp):
    data1D = images.data1D[idx]
    # notice data1D here is centerlized ( mean = 0 along pixel direction)
    pca_mat = np.dot(data1D,pc_eigens)
    
    images.pca_mat[idx] = pca_mat
#     print('pca_mat',pca_mat)
    '''
    mapcode (n_comp, pixel) - image to (1, pixel) - image with base specified in input
    '''
    min_max = preprocessing.MinMaxScaler(feature_range=(0, base-1))
        # X array-like of shape (n_samples, n_features)
    stream = min_max.fit_transform(pca_mat)
    stream = np.trunc(stream).T
    print('rebined stream shape',np.shape(stream))
    images.stream[idx] = stream
    mapcode = 0
    for comp in range(pca_comp):
        mapcode = mapcode + stream[comp]*base**comp
#      print(mapcode, self.stream_1Ds)
    stream_1D = mapcode.astype(int)
    print(f'No.{idx} 1D stream prepared')
#     np.savetxt(f'./UMAP\\stream_test\\{idx}.txt',stream_1D)
    images.stream1D[idx] = stream_1D
    return stream_1D
        
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
    num = 2
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
        stream_concat = process_dataset_test(data_dir, base, num, pca_comp, topk, test)
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
    
    



    
#     convert = Convert(images, base = base,  pca_comp =  pca_comp)
#     convert.prepare(idx)
#     print(images.pca)

#def parallel_process(images, base):   
#    with ThreadPoolExecutor() as executor:
#        print(images.idxs)
#        # Get a list of files to process
#        # Process the list of files, but split the work across the process pool to use all CPUs! 
#        futures = []
#        print('Creating Convert Object for PCA Preprocessing')
##        convert = Convert(images, base)
#        print('converting')
#        for idx in images.idxs:
#            futures.append(executor.submit(lambda x: par_convert(x, images, base = 4, topk = 10), idx))
#            print(f" No.{idx} submitted for PCA convertion")
#        
#        for future in as_completed(futures):
#            future.result
#            
#        print(future.done())             
#        naivecount = future.result()
##        print(future.done())
#    return naivecount

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


#os.environ["PYTHONPATH"] = r'C:\Users\viska\.conda\envs\streaming\python.exe'
#sys.path.append( r'C:\Users\viska\.conda\envs\streaming\python.exe')


    
# def run_step_convert(idx, images, base,  pca_comp, test = 'False'):
#     if test:
#         images.get_test_data(idx)
#     else:
#         images.get_data(idx)
#     convert = Convert(images, base = base,  pca_comp =  pca_comp)
#     convert.prepare(idx)
#     print(f'No.{idx} 1D stream prepared')
#     return convert

# def run_step_concatenate(converts,base):
#     stream_concat = []
#     for convert in converts:
#         stream_1D = convert.stream_1D
#         stream_concat = np.concatenate((stream_concat, stream_1D), axis=0) 
# #    np.savetxt(f'stream_concat_b{base}.txt', stream_concat)
#     return stream_concat

# def run_step_naivecount(stream_concat, topk):
#     exactcount = ExactCount(stream_concat, topk = topk)
#     print('##################################Run step topk ###################################')
#     exactcount.run_step_topk()
#     exact_HH = exactcount.topk_sort
# #    print(f'##################################Exact Count Results###################################')
# #    print ("exact_hhs",exact_hhs[:display])
# #    print ("exact_freq", exact_freq[:display])   
#     return exact_HH


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

# def process_exact_count(converts, base, topk , name): 
# #    print('##################################Running Exact Count###################################')
#     # start the timer for exact counting
#     start_exact = timer()
#     # concatenating all images 1D stream into a long 1D stream
#     print('##################################Run step concatenate ###################################')
#     stream_concat = run_step_concatenate(converts, base)
#     print("stream_concat of size",np.shape(stream_concat),stream_concat)
#     np.savetxt(f'{name}\\stream_concat.csv', stream_concat) 
#     print('##################################Run step naive count ###################################')
#     exact_HH = run_step_naivecount(stream_concat, topk = topk)
#     end_exact = timer()
#     exact_time = end_exact - start_exact
#     return  exact_HH, exact_time

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

# def finish(exact_time, sketch_times):          
#     print("Time took for exact vs sketch counting", round(exact_time,3), 'vs',sketch_times)
# #    print("Count Sktech Time is faster", exact_time - sketch_times)
#     return None 

# def process_dataset1(DATASET_PATH, base, num, pca_comp, test):   
#     images = dataset(DATASET_PATH = DATASET_PATH)
#     images.get_data_path(num = num)
#     pca_combined = np.zeros([images.layer, images.layer])
#     print(f'ThreadPoolExecutor Initiated for {images.N_img} images with size {images.size} ')
#     with ThreadPoolExecutor() as executor: 
#         futures = []
#         for idx in images.idxs:
#             futures.append(executor.submit(lambda x: run_step_covariance(x, images, test), idx))
#             print(f" No.{idx} image is loaded")
#         cov_combined = np.zeros([images.layer,images.layer])
#         for future in as_completed(futures):
#             result = future.result()
#             cov_combined += result
#             print('processed result',result[:5,:5])
#         print("Done?", future.done())
# #         for val in result.values():
# #             pca_combined += val
#         print(cov_combined)
#         pc_eigens = images.get_pc_eigens(cov_combined, pca_comp)
# #         for val images.data.items()
#     print('pc_eigens',pc_eigens[0:2,:], np.shape(pc_eigens))
#     return pc_eigens, images

# def process_pca(pc_eigens, images, base, pca_comp):
#     with ThreadPoolExecutor() as executor: 
#         futures = []
#         for idx in images.idxs:
#             futures.append(executor.submit(lambda x: run_step_pca(x, pc_eigens, images, base, pca_comp), idx))
#         stream_1Ds = []
#         for future in as_completed(futures):
#             result = future.result()
#             stream_1Ds = np.append(stream_1Ds,result)
# #             print('processed result',result[[0,1,-2,-1]])
#         print("Done?", future.done())
#     print(np.shape(stream_1Ds))
#     print(stream_1Ds)
#     return stream_1Ds
# def run_exact_count(stream_concat, topk):
#     c = Counter(stream_concat)
# #     exactcount = ExactCount(stream_concat, topk = topk)
#     print(f'################################## Naive counting topk {topk} ###################################')
# #     exactcount.get_topk()
#     exact_HH = c.most_common(topk) 
#     return exact_HH  