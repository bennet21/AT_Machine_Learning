# data
import netCDF4 as nc
import numpy as np

import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn.metrics import mean_squared_error
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from scipy.spatial import cKDTree

from scipy.ndimage.interpolation import zoom

## ----- DATA PREP ---------

def regrid(x,K):
    '''
    Function to regrid a dataset with a factor K, this is useful since to compute performance statistics the datasets need to have the same shape.
    '''
    m1 = 0
    m2 = 0
    arr = np.zeros((x.shape[0],x.shape[1]*K,x.shape[2]*K), dtype='float32')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if j % K == 0:
                    # j is new modulo zero keep using this value for j
                    m1 = int(j / K)
                    #print(m1)
                if k % K == 0:
                    m2 = int(k / K)
                arr[i,j,k] = x[i,m1,m2]
    return arr

def lon_lat_to_cartesian(lon, lat):
    '''
        Function to transform longitude and latitude to a cartesian system, based on WGS 84 coordinate reference system. Inputs need to be 2D, if they are 1D you can transform them with np.meshgrid().
        inputs:
            lon - longitude
            lat - latitude
            
        
        outputs: x,y,z cartesian coordinates of the lat,lon coordinates
    '''
    # WGS 84 reference coordinate system parameters
    A = 6378.137 # major axis [km]   
    E2 = 6.69437999014e-3 # eccentricity squared 
    
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)
    return x,y,z

def transform_extent(data, inds, shape):
    '''
    
    '''    
    # loops over dataset and takes the subset defined by the indices and shape
    transformed_data = []
    
    if(len(data.shape)>2):
        for d in data:
            transformed_data.append(np.array(d.flatten()[inds].reshape(shape)))
    else:
        transformed_data = data.flatten()[inds].reshape(shape)
        
    return np.array(transformed_data)

def minmax_norm(data):
    '''
        Min-max normalises a dataset between range [0,1].
    '''
    max = np.max(data)
    min = np.min(data)
    
    arr = np.zeros(shape = data.shape, dtype="float32")
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            arr[i,j] = (data[i,j] - min) / (max - min)
    return(arr)

def downscale_image(x, K):
    '''
        D
    '''
    if x.ndim == 3:
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2],1))
        
    ds_out = np.zeros(shape = (x.shape[0], int(x.shape[1]/K), int(x.shape[2]/K), x.shape[3]), dtype="float32")
    
    for j in range(x.shape[3]):
        for i in range(x.shape[0]):
            ds_out[i,:,:,j] = zoom(x[i,:,:,j],1/K)

    return ds_out

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_TFRecords(filename, data_HR=None, data_LR=None, mode='test'):
    '''
        Generate TFRecords files for model training or testing

        inputs:
            filename - filename for TFRecord (should by type *.tfrecord)
            data     - numpy array of size (N, h, w, c) containing data to be written to TFRecord
            model    - if 'train', then data contains HR data that is coarsened k times 
                       and both HR and LR data written to TFRecord
                       if 'test', then data contains LR data 

        outputs:
            No output, but .tfrecord file written to filename
    '''
    if data_HR is not None and data_HR.ndim == 3:
        data_HR = data_HR.reshape((data_HR.shape[0],data_HR.shape[1],data_HR.shape[2],1))
        
    if data_LR.ndim == 3:
        data_LR = data_LR.reshape((data_LR.shape[0],data_LR.shape[1],data_LR.shape[2],1))

    with tf.io.TFRecordWriter(filename) as writer:
        for j in range(data_LR.shape[0]):
            if mode == 'train':
                h_HR, w_HR, c = data_HR[j, ...].shape
                h_LR, w_LR, c = data_LR[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                   'data_HR': _bytes_feature(data_HR[j, ...].tostring()),
                                      'h_HR': _int64_feature(h_HR),
                                      'w_HR': _int64_feature(w_HR),
                                         'c': _int64_feature(c)})
            elif mode == 'test':
                h_LR, w_LR, c = data_LR[j, ...].shape
                features = tf.train.Features(feature={
                                     'index': _int64_feature(j),
                                   'data_LR': _bytes_feature(data_LR[j, ...].tostring()),
                                      'h_LR': _int64_feature(h_LR),
                                      'w_LR': _int64_feature(w_LR),
                                         'c': _int64_feature(c)})

            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

## ----- EDA ---------

def rms(A,B):
    return mean_squared_error(A, B, squared=False)  

def moving_average(arr, window_size):
    '''
        Calculates the moving average of the given array using the supplied window size.
    '''
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:

        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i : i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1
    
    return(moving_averages)

def rescale(data,max,min):
    '''
        To transform min-max normalised data back to their original scale.
        inputs:
            max - original max
            min - original min
            
        outputs:
            rescaled dataset
    '''
    arr = np.zeros(shape = data.shape, dtype="float32")
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            arr[i,j] = min+data[i,j]*(max-min) 
    return(arr)

def performance_dataset(A,B, mode):
    '''
        Calculates performance measures RMSE and SSIM between datasets A and B.
    '''
    rms_arr = []
    ssim_arr = []
    for t in range(A.shape[0]):
        rms_arr.append(rms(A[t],B[t]))
        ssim_arr.append(ssim(A[t],B[t]))
        
    rms_arr = np.array(rms_arr)
    ssim_arr = np.array(ssim_arr)
    if mode == 'array':
        return (rms_arr, ssim_arr)
    if mode == 'mean':
        return (np.mean(rms_arr),np.mean(ssim_arr))
    else:
        return None


def extract_performance(model_path,data_train,data_test,c=0,minmax=False,max_epoch=32,Regrid=True):
    '''
    Calculates SSIM and RMSE performance for SR and saves it to a dataframe in csv format
    '''
    
    epoch = []
    
    SSIM_train = []
    SSIM_test = []

    rms_train = []
    rms_test = []

    
    for subdir, dirs, files in os.walk(model_path+'/train'):
        dirs.sort()
        if ('cnn' in subdir or 'gan' in subdir) and int(subdir[-2:]) <= max_epoch:
            print(subdir)
            epoch.append(int(subdir[-2:]))
            if(np.load(subdir + '/dataSR.npy').ndim==4 and Regrid):
                r = regrid(np.load(subdir + '/dataSR.npy')[:,:,:,c],3)
            elif Regrid:
                r = regrid(np.load(subdir + '/dataSR.npy'),3)
            else:
                r = np.load(subdir + '/dataSR.npy')[:,:,:,c]
            if minmax:
                r = rescale(r,np.max(data_train),np.min(data_train))
            p = performance_dataset(r,data_train,'mean')
            rms_train.append(p[0])
            SSIM_train.append(p[1])
        
        
    for subdir, dirs, files in os.walk(model_path+'/test'):
        dirs.sort()
        if ('cnn' in subdir or 'gan' in subdir) and int(subdir[-2:]) <= max_epoch:
            print(subdir)
            if(np.load(subdir + '/dataSR.npy').ndim==4) and Regrid:
                r = regrid(np.load(subdir + '/dataSR.npy')[:,:,:,c],3)
            elif Regrid:
                r = regrid(np.load(subdir + '/dataSR.npy'),3)
            else:
                r = np.load(subdir + '/dataSR.npy')[:,:,:,c]
            if minmax:
                r = rescale(r,np.max(data_train),np.min(data_train))
            p = performance_dataset(r,data_test,'mean')
            rms_test.append(p[0])
            SSIM_test.append(p[1])
    
    performance_df = pd.DataFrame()

    for i in range(len(epoch)):
        vals = {'Epoch': '%.3f' % epoch[i], 
                'SSIM_train': SSIM_train[i],
                'SSIM_test': SSIM_test[i],
                'rms_train': rms_train[i],
                'rms_test': rms_test[i]}

        performance_df = performance_df.append(vals, ignore_index = True)

    performance_df.to_csv(model_path+'/performance_df_'+str(c)+'.csv', index=False)
    
    
def performance_map(data_HR,data_LR):
    '''
    calculates the maps of the RMSE and SSIM stats
    
    mode: rms or ssim
    '''
    rms_map = np.zeros(shape=(data_HR.shape[1],data_HR.shape[2]), dtype='float32')
    ssim_map = np.zeros(shape=(data_HR.shape[1],data_HR.shape[2]), dtype='float32')
    
    for i in range(data_HR.shape[1]):
        for j in range(data_HR.shape[2]):
            rms_map[i,j] = rms(data_HR[:,i,j],data_LR[:,i,j])
            ssim_map[i,j] = ssim(data_HR[:,i,j],data_LR[:,i,j])
    
    return(rms_map, ssim_map)

def performance_time(data_HR, data_LR):
    '''
    calculates the timeseries of the RMSE and SSIM stats
    
    mode: rms or ssim
    '''
    rms_arr = []
    ssim_arr = []
    
    for t in range(data_HR.shape[0]):
        rms_arr.append(rms(data_HR[t],data_LR[t]))
        ssim_arr.append(ssim(data_HR[t],data_LR[t]))
    
    return(rms_arr, ssim_arr)


    
        

    
    
    
