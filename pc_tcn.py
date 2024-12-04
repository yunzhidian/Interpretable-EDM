import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import random
import numpy as np
import os
import sys


def build_tcn_v5(n_timesteps, n_features, n_classes=6, 
              n_units=32, k_size=5, n_stacks=1, dilation=(1,2,4,8), dropout=0.5,
              norm_type='layer_norm', activation_type='relu',trasit_type='average',
              isMdlPlot=True):
    """
    Final version allowing for regulation of hyperparameters
    """
    # first get the current directory, then the parent directory and add it to the env
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'lib/')))
    # from attention import Attention
    from my_attention import Attention

    from keras.layers.merge import add
    receptive_size= 1+2*(k_size-1)*n_stacks*sum(dilation)
    print(f'receptive field size:{receptive_size}')

    set_seed()
    # set the activation type
    def activation_layer(activation_type, x):
        if activation_type=='relu':
            activation_layer= layers.ReLU()(x)
        elif activation_type=='leaky_relu':
            activation_layer= layers.LeakyReLU(alpha=0.3)(x)
        elif activation_type=='elu':
            activation_layer= layers.ELU(alpha=0.1)(x)
        elif activation_type=='gelu':
            # activation_layer= layers.Lambda(keras.activations.gelu)(x)
            activation_layer= layers.Activation(keras.activations.gelu)(x)
        else:
            raise ValueError('Not supported activation function!')
        return activation_layer
    # set the normal layer type
    def norm_layer(norm_type, x):
        if norm_type=='layer_norm':
            norm_layer= layers.LayerNormalization()(x)
        elif norm_type=='batch_norm':
            norm_layer= layers.BatchNormalization()(x)
        else:
            raise ValueError('Not supported normal layer type!')
        return norm_layer

    # Functional API
    inputs= layers.Input(shape=(n_timesteps, n_features))
    # Input embedding
    x_in= layers.Dense(units=n_units)(inputs)
    x= norm_layer(norm_type, x_in)
    x= layers.Dropout(dropout)(x)
    x= activation_layer(activation_type, x)

    # residual block
    for k in range(n_stacks):
        x_tcn= x
        for i, d in enumerate(dilation):
            x_tcn= layers.Conv1D(filters=n_units, kernel_size=k_size, dilation_rate=d, 
                                 padding='same', name='tcn1_conv_'+str(k)+'_'+str(i))(x_tcn)
        x_tcn= norm_layer(norm_type, x_tcn)
        x_tcn= activation_layer(activation_type, x_tcn)
        for i, d in enumerate(dilation):
            x_tcn= layers.Conv1D(filters=n_units, kernel_size=k_size, dilation_rate=d, 
                                 padding='same', name='tcn2_conv_'+str(k)+'_'+str(i))(x_tcn)
        x_tcn= norm_layer(norm_type, x_tcn)
        x = add([x, x_tcn])
        x= activation_layer(activation_type, x)

    if trasit_type=='average':
        # 1) global average
        x= layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    elif trasit_type=='slice_last':
        # 2) take the last 
        def slice_last(x):
            return x[..., -1, :]
        x = layers.Lambda(slice_last)(x)
    elif trasit_type=='attention':
        # 3) attention mechanism
        x,_= Attention(n_units)(x)
    else:
        raise ValueError('Not supported transit type!')

    # classifier
    x= layers.Dense(n_classes, activation='softmax', name='tcn_sfm')(x)
    tcn_mdl=keras.Model(inputs, x, name="tcn_model")
    if isMdlPlot:
        tcn_mdl.summary()
        plot_model(tcn_mdl, 'mdl_pics/tcn_v5.png', show_shapes=True)

    return tcn_mdl


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")