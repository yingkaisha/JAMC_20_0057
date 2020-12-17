
import sys
import time
import argparse
from glob import glob

import h5py
import numpy as np
from random import shuffle
from tensorflow import keras

# custom tools
sys.path.insert(0, '/your-path-of-repo/utils/')
from namelist import *
import model_utils as mu
import train_utils as tu

# ---------------------------------------------------------- #

# parse user inputs
parser = argparse.ArgumentParser()

# one of the  'TMAX', 'TMIN', 'TMEAN', 'PCT'
parser.add_argument('v', help='Downscaling variable name')

# one of the 'annual', 'summer', 'winter', 'djf', 'mam', 'jja', 'son'
parser.add_argument('s', help='Training seasons')

# a number of 1, 2, ..., 5
# but 3 is expected, the rest are trial-and-error options
parser.add_argument('c1', help='Number of input channels (1<c<5)')

# 2 is expected
parser.add_argument('c2', help='Number of output channels (1<c<3)')
args = vars(parser.parse_args())

# parser handling
VAR, seasons, input_flag, output_flag = tu.parser_handler(args)
N_input = int(np.sum(input_flag))
N_output = int(np.sum(output_flag))

# ---------------------------------------------------------- #

# number of filters based on the downscaling variable
if VAR == 'PCT':
    print('PCT hidden layer setup')
    N = [64, 96, 128, 160]
else:
    print('T2 hidden layer setup')
    N = [56, 112, 224, 448]

# ---------------------------------------------------------- #

lr = 5e-5
epochs = 150
activation='relu'
pool=False # stride convolution instead of maxpooling

# early stopping settings
min_del = 0
max_tol = 3 # early stopping with patience

# ---------------------------------------------------------- #
# training by seasons
for sea in seasons:
    
    # UAE configuration
    dscale_uae = mu.UNET_AE(N, (None, None, N_input), pool=pool, activation=activation)
    opt_ = keras.optimizers.Adam(lr=lr)
    dscale_uae.compile(loss=keras.losses.mean_absolute_error, optimizer=opt_)
    
    # check point settings
    # "temp_dir" is where models are saved, defined in the namelist.py
    save_name = 'UAE_raw_{}_{}'.format(VAR, sea)
    save_path = temp_dir+save_name+'/'
    hist_path = temp_dir+'{}_loss.npy'.format(save_name)

    # allocate arrays for training/validation loss
    T_LOSS = np.empty((int(epochs*L_train), 2)); T_LOSS[...] = np.nan
    V_LOSS = np.empty((epochs,)); V_LOSS[...] = np.nan
    
    # ---------------------- data section ---------------------- #

    # before training starts, the authors have saved individual 
    # training batches separatly as numpy files, and the entire 
    # validation set as a hdf5 file 
    # "input_flag" and "output_flag" are boolean indices that 
    # subset input and training slices from the saved numpy arrays
    
    # Size of data is out of repo capacity, so pesudo code is
    # provided here
    # ---------------------------------------------------------- #
    trainfiles = glob('/user/drive/dscal_proj/batch_train_{}_{}_*.npy'.format(VAR, sea))
    validfile = glob('/user/drive/dscal_proj/valid_{}_{}_*.npy'.format(VAR, sea))
    
    with h5py.File(validfile, 'r') as h5io:
        valid_all = h5io['valid'][...] # shape = (sample, x, y, channel)
    X_valid = valid_all[..., input_flag]
    Y_valid = valid_all[..., output_flag]
    
    #
    shuffle(trainfiles)
    L_train = len(trainfiles)

    # epoch begins
    tol = 0              
    for i in range(epochs):
        print('epoch = {}'.format(i))
        if i == 0:
            Y_pred = dscale_uae.predict([X_valid])
            # validate HR temp branch only
            record = tu.mean_absolute_error(Y_pred[..., 0], Y_valid[..., 0])
            print('Initial validation loss: {}'.format(record))

        # shuffling on epoch-begin
        shuffle(trainfiles)
        
        # loop over batches
        for j, name in enumerate(trainfiles): 
            
            # import batch data (numpy arrays)
            temp_batch = np.load(name)
            X = temp_batch[..., input_flag]
            Y = temp_batch[..., output_flag]
            # temp_batch.shape = (sample_per_batch, x, y, channel)
            
            # train on batch
            loss_ = dscale_uae.train_on_batch(X, Y)
            
            # Backup training loss
            T_LOSS[i*L_train+j, :] = loss_
            
            # print out
            if j%50 == 0:
                print('\t{} step temp loss = {}, elev loss = {}'.format(j, loss_[0], loss_[1]))
                
        # validate on epoch-end
        Y_pred = dscale_uae.predict([X_valid])
        # validate HR temp branch only
        record_temp = tu.mean_absolute_error(Y_pred[..., 0], Y_valid[..., 0])

        # Backup validation loss
        V_LOSS[i] = record_temp
        
        # Save loss info
        loss_dict = {'T_LOSS':T_LOSS, 'V_LOSS':V_LOSS}
        np.save(hist_path, loss_dict)
        
        # early stopping
        if record - record_temp > min_del:
            print('Validation loss improved from {} to {}'.format(record, record_temp))
            record = record_temp
            tol = 0
            print('tol: {}'.format(tol))
            # save
            print('save to: {}'.format(save_path))
            dscale_uae.save(save_path)
        else:
            print('Validation loss {} NOT improved'.format(record_temp))
            tol += 1
            print('tol: {}'.format(tol))
            if tol >= max_tol:
                print('Early stopping')
                sys.exit();
            else:
                print('Pass to the next epoch')
                continue;
