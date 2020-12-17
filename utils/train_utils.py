import sys
import glob
import numpy as np
from tensorflow import keras


def mean_absolute_error(X, Y):
    '''
    mean absolute error
    nan is not expected
    '''
    return np.mean(np.abs(X-Y))

def dummy_loader(model_path):
    '''
    A function that imports pre-trained model weights
    '''
    print('Import model:\n{}'.format(model_path))
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

def parser_handler(args):
    '''
    Convert argparse positional dictionary into variables
    Specified for downscaling model training
    '''
    args['c1'] = int(args['c1'])
    args['c2'] = int(args['c2'])
    
    # arguments handling
    if args['v'] in ['TMAX', 'TMIN', 'TMEAN', 'PCT']:
        VAR = args['v']
    else:
        raise ValueError('Wrong variable')

    if args['s'] == 'annual':
        seasons = ['djf', 'mam', 'jja', 'son']
    elif args['s'] == 'summer':
        seasons = ['jja', 'son']
    elif args['s'] == 'winter':
        seasons = ['djf', 'mam']
    elif args['s'] in ['djf', 'mam', 'jja', 'son']:
        seasons = [args['s']]
    else:
        raise ValueError('Wrong season')

    if args['c1'] == 1:
        input_flag = [False, True, False, False, False, False]
    elif args['c1'] == 2:
        input_flag = [False, True, False, False, True, False]
    elif args['c1'] == 3:
        input_flag = [False, True, False, False, True, True]
    elif args['c1'] == 4:
        input_flag = [False, True, True, False, True, True] 
    elif args['c1'] == 5:
        input_flag = [False, True, True, True, True, True]
    else:
        raise ValueError('Wrong input channel numbers')
    
    if args['c2'] == 1:
        output_flag = [True, False, False, False, False, False]
    elif args['c2'] == 2:
        output_flag = [True, False, False, False, True, False]
    elif args['c2'] == 3:
        output_flag = [True, True, False, False, True, False]
    else:
        raise ValueError('Wrong target numbers')
    
    return VAR, seasons, input_flag, output_flag
