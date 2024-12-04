import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from keras.layers.merge import add
from keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import random
import numpy as np
import os
import sys

def build_residual_lstm_v5(n_timesteps, n_features, n_classes=6, 
                           n_units=32, n_layers=3, dropout=0.5, 
                           activation_type='relu', norm_type='layer_norm', trasit_type='average',
                           isMdlPlot=True):
    """
    Final version allowing for regulation of hyperparameters
    """
    # first get the current directory, then the parent directory and add it to the env
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'lib/')))
    # from attention import Attention
    from my_attention import Attention

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
            activation_layer= layers.Activation(keras.activations.gelu)(x)
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
    x_in= layers.Dense(units=n_units*2)(inputs)
    x= norm_layer(norm_type, x_in)
    x= layers.Dropout(dropout)(x)
    x= activation_layer(activation_type, x)

    # residual block
    for i in range(n_layers):
        x_lstm= layers.Bidirectional(
                layers.LSTM(units=n_units, name='lstm_units_'+str(i+1)+'_d1',
                            kernel_regularizer=l2(3e-2), recurrent_regularizer=l2(3e-2),
                            return_sequences=True)
        )(x)
        x_lstm= norm_layer(norm_type, x_lstm)
        x_lstm= activation_layer(activation_type, x_lstm)
        x_lstm= layers.Bidirectional(
                layers.LSTM(units=n_units, name='lstm_units_'+str(i+1)+'_d2',
                            kernel_regularizer=l2(3e-2), recurrent_regularizer=l2(3e-2),
                            return_sequences=True)
        )(x_lstm)
        x_lstm= norm_layer(norm_type, x_lstm)
        # skip connection
        x = add([x, x_lstm])
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
        x,_= Attention(units=n_units*2)(x)
    else:
        raise ValueError('Not supported transit type!')
    
    # classifier
    x= layers.Dense(n_classes, activation='softmax', name='residual_lstm__sfm' )(x)
    lstm_mdl=keras.Model(inputs, x, name="residual_lstm_model")
    if isMdlPlot:
        lstm_mdl.summary()
        plot_model(lstm_mdl, 'mdl_pics/residual-lstm.png', show_shapes=True)

    return lstm_mdl


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