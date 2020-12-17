import glob
import random
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K

def stride_conv(X, channel, pool_size=2, activation='relu', name='X'):
    '''
    stride convolutional layer --> batch normalization --> Activation
    *Proposed to replace max- and average-pooling layers 
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        pool_size: size of stride
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor
    '''
    # linear convolution with strides
    X = keras.layers.Conv2D(channel, pool_size, strides=(pool_size, pool_size), padding='valid', 
                            use_bias=False, kernel_initializer='he_normal', name=name+'_stride_conv')(X)
    # batch normalization
    X = keras.layers.BatchNormalization(axis=3, name=name+'_stride_conv_bn')(X)
    
    # activation
    if activation == 'relu':
        X = keras.layers.ReLU(name=name+'_stride_conv_relu')(X)
    elif activation == 'leaky':
        X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_stride_conv_leaky')(X)
        
    return X

def CONV_stack(X, channel, kernel_size=3, stack_num=2, activation='relu', name='conv_stack'):
    '''
    (Convolutional layer --> batch normalization --> Activation)*stack_num
    
    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        stack_num: number of stacked convolutional layers
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor
        
    '''
    
    # stacking Convolutional layers
    for i in range(stack_num):
        
        # linear convolution
        X = keras.layers.Conv2D(channel, kernel_size, padding='same', use_bias=False, 
                                kernel_initializer='he_normal', name=name+'_stack{}_conv'.format(i))(X)
        
        # batch normalization
        X = keras.layers.BatchNormalization(axis=3, name=name+'_stack{}_bn'.format(i))(X)
        
        # activation
        if activation == 'relu':
            X = keras.layers.ReLU(name=name+'_stack{}_relu'.format(i))(X)
        elif activation == 'leaky':
            X = keras.layers.LeakyReLU(alpha=0.3, name=name+'_stack{}_leaky'.format(i))(X)
            
    return X


def UNET_left(X, channel, kernel_size=3, pool_size=2, pool=True, activation='relu', name='left0'):
    '''
    Encoder block of UNet

    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        pool_size: size of stride
        pool: True for maxpooling, False for strided convolutional layers
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor
        
    '''
    
    # maxpooling layer vs strided convolutional layers
    if pool:
        X = keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name='{}_pool'.format(name))(X)
    else:
        X = stride_conv(X, channel, pool_size, activation=activation, name=name)
        
    # stack linear convolutional layers
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, name=name)
    
    return X

def UNET_right(X, X_left, channel, kernel_size=3, pool_size=2, activation='relu', name='right0'):
    '''
    Decoder block of UNet

    Input
    ----------
        X: input tensor
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        pool_size: size of transpose stride, expected to be the same as "UNET_left"
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor
        
    '''
    
    # Transpose convolutional layer --> stacked linear convolutional layers
    X = keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                                     padding='same', name=name+'_trans_conv')(X)
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation, name=name+'_conv_after_trans')
    
    # Tensor concatenation
    H = keras.layers.concatenate([X_left, X], axis=3)
    
    # stacked linear convolutional layers after concatenation
    H = CONV_stack(H, channel, kernel_size, stack_num=1, activation=activation, name=name+'_conv_after_concat')
    
    return H

def XNET_right(X_conv, X_list, channel, kernel_size=3, pool_size=2, activation='relu', name='right0'):
    '''
    Decoder block of Nest-UNet

    Input
    ----------
        X: input tensor
        X_list: a list of other corresponded input tensors (see Sha 2020b, Figure 2) 
        channel: number of convolution filters
        kernel_size: size of 2-d convolution kernels
        pool_size: size of transpose stride, expected to be the same as "UNET_left"
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        name: name of created keras layers

    Output
    ----------
        X: output tensor
        
    '''
    
    # Transpose convolutional layer --> concatenation
    X_unpool = keras.layers.Conv2DTranspose(channel, kernel_size, strides=(pool_size, pool_size), 
                                            padding='same', name=name+'_trans_conv')(X_conv)
    
    # <--- *stacked convolutional can be applied here
    X_unpool = keras.layers.concatenate([X_unpool]+X_list, axis=3, name=name+'_nest')
    
    # Stacked convolutions after concatenation 
    X_conv = CONV_stack(X_unpool, channel, kernel_size, stack_num=2, activation=activation, name=name+'_conv_after_concat')
    
    return X_conv

def UNET(layer_N, input_size, input_stack_num=2, pool=True, activation='relu'):
    '''
    UNet with three down- and upsampling levels.
    
    Input
    ----------
        layer_N: Number of convolution filters on each down- and upsampling levels
                 Should be an iterable of four int numbers, e.g., [32, 64, 128, 256]
        
        input_size: a tuple that feed input keras.layers.Input, e.g. (None, None, 3)
        
        input_stack_num: number of stacked convolutional layers before entering the first downsampling block
        
        pool: True for maxpooling, False for strided convolutional layers
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU

    Output
    ----------
        model: the Keras functional API model, i.e., keras.models.Model(inputs=[IN], outputs=[OUT])
    
    '''
    
    # input layer
    IN = keras.layers.Input(input_size, name='unet_in')
    X_en1 = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
    
    # downsampling levels
    X_en2 = UNET_left(X_en1, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X_en3 = UNET_left(X_en2, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    X4 = UNET_left(X_en3, layer_N[3], pool=pool, activation=activation, name='unet_bottom')
    
    # upsampling levels
    X_de3 = UNET_right(X4, X_en3, layer_N[2], activation=activation, name='unet_right2')
    X_de2 = UNET_right(X_de3, X_en2, layer_N[1], activation=activation, name='unet_right1')
    X_de1 = UNET_right(X_de2, X_en1, layer_N[0], activation=activation, name='unet_right0')
    
    # output
    OUT = CONV_stack(X_de1, 2, kernel_size=3, stack_num=1, activation=activation, name='unet_out')
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, padding='same', name='unet_exit')(OUT)
    
    # model
    model = keras.models.Model(inputs=[IN], outputs=[OUT])
    
    return model

def UNET_AE(layer_N, input_size, input_stack_num=2, pool=True, activation='relu', drop_rate=0.05):
    '''
    UNet-AE with three down- and upsampling levels.
    
    Input
    ----------
        layer_N: Number of convolution filters on each down- and upsampling levels
                 Should be an iterable of four int numbers, e.g., [32, 64, 128, 256]
        
        input_size: a tuple that feed input keras.layers.Input, e.g. (None, None, 3)
        input_stack_num: number of stacked convolutional layers before entering the first downsampling block
        activation: 'relu' for ReLU or 'leaky' for Leaky ReLU
        pool: True for maxpooling, False for strided convolutional layers
        drop_rate: *optional spatial dropout before upsampling

    Output
    ----------
        model: the Keras functional API model, i.e., keras.models.Model(inputs=[IN], outputs=[OUT])
    
    '''
    if drop_rate <= 0:
        drop_flag = False
    else:
        drop_flag = True
        
    # input layer
    IN = keras.layers.Input(input_size)
    X_en1 = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='uae_left0')
    
    # downsampling levels
    X_en2 = UNET_left(X_en1, layer_N[1], pool=pool, activation=activation, name='uae_left1')
    X_en3 = UNET_left(X_en2, layer_N[2], pool=pool, activation=activation, name='uae_left2')
    X4 = UNET_left(X_en3, layer_N[3], pool=pool, activation=activation, name='uae_bottom')
    
    # dropout at the bottom of "U"
    if drop_flag:
        X4 = keras.layers.SpatialDropout2D(drop_rate)(X4)
        
    # upsampling levels
    X_de3 = UNET_right(X4, X_en3, layer_N[2], activation=activation, name='uae_right2')
    X_de2 = UNET_right(X_de3, X_en2, layer_N[1], activation=activation, name='uae_right1')
    X_de1 = UNET_right(X_de2, X_en1, layer_N[0], activation=activation, name='uae_right0')
        
    # output (supervised, HR downscaling target)
    OUT1 = CONV_stack(X_de1, 2, kernel_size=3, stack_num=1)
    OUT1 = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, name='HR_temp')(OUT1)
    
    # output (unsupervised, HR elevation)
    OUT2 = CONV_stack(X_de1, 2, kernel_size=3, stack_num=1)
    OUT2 = keras.layers.Conv2D(1, 1, activation=keras.activations.linear, name='HR_elev')(OUT2)
    
    model = keras.models.Model(inputs=[IN], outputs=[OUT1, OUT2])
    
    return model

def XNET(layer_N, input_size, input_stack_num=2, pool=True, activation='relu'):
    '''
    UNet++ with 8x downsampling rate
    '''
    
    # input layer
    IN = keras.layers.Input(input_size)
    X11_conv = CONV_stack(IN, layer_N[0], kernel_size=3, stack_num=input_stack_num, activation=activation, name='unet_left0')
    
    # downsampling levels (same as in the UNET)
    X21_conv = UNET_left(X11_conv, layer_N[1], pool=pool, activation=activation, name='unet_left1')
    X31_conv = UNET_left(X21_conv, layer_N[2], pool=pool, activation=activation, name='unet_left2')
    X41_conv = UNET_left(X31_conv, layer_N[3], pool=pool, activation=activation, name='unet_bottom')
    
    # up-sampling part 1
    X12_conv = XNET_right(X21_conv, [X11_conv], layer_N[0], activation=activation, name='xnet_12') 
    X22_conv = XNET_right(X31_conv, [X21_conv], layer_N[1], activation=activation, name='xnet_22')
    X32_conv = XNET_right(X41_conv, [X31_conv], layer_N[2], activation=activation, name='xnet_32')
    
    # up-sampling part 2
    X13_conv = XNET_right(X22_conv, [X11_conv, X12_conv], layer_N[0], activation=activation, name='xnet_right13') 
    X23_conv = XNET_right(X32_conv, [X21_conv, X22_conv], layer_N[1], activation=activation, name='xnet_right23')
    
    # up-sampling part 3
    X14_conv = XNET_right(X23_conv, [X11_conv, X12_conv, X13_conv], layer_N[0], activation=activation, name='xnet_right14')
    
    # output
    OUT = CONV_stack(X14_conv, 2, kernel_size=3, stack_num=1, activation=activation, name='xnet_out')
    OUT = keras.layers.Conv2D(1, 1, activation=keras.activations.linear)(OUT)
    
    # model
    model = keras.models.Model(inputs=[IN], outputs=[OUT])
    
    return model
