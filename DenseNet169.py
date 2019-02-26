from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, AveragePooling2D, GlobalAveragePooling2D, Concatenate
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras
import keras.backend as K


def TransitionBlock(x,nb_filters,dropout_rate=None,weight_decay):
    x = BatchNormalization()(x)
    x = Conv2D(nb_filters,(1,1),padding='same')
    x = GlobalAveragePooling2D(pool_size=(2,2),strides=(2,2))

    return x

def ConvBlock(x,nb_filters,dropout_rate=None,weight_decay):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filters,(1,1),padding='same')
    x = Conv2D(nb_filters,(3,3),padding='same')

def DenseBlock(x,nb_layers,nb_filters,growth_rate,dropout_rate=None,weight_decay):
    for index in range(nb_layers):
        x = ConvBlock(x,nb_filters,dropout_rate=None,weight_decay)
        nb_filters+=growth_rate

    return x, nb_filters


def DenseNet169(nb_classes,img_dim,depth,nb_dense_block,
    growth_rate,nb_filters,dropout_rate=None,weight_decay):
    model_input = Input(shape=img_dim)
    x = Conv2D(nb_filters,(3,3),padding='same',kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_id in range(nb_dense_block-1):
        x, nb_filters = DenseBlock(x,nb_layers,nb_filters,growth_rate,dropout_rate=None,weight_decay)
        x = TransitionBlock(x,nb_filters,dropout_rate=None,weight_decay)

    # adding the last dense block
    x, nb_filters = DenseBlock(x,nb_layers,nb_filters,growth_rate,dropout_rate=None,weight_decay)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(pool_size=(7,7))(x)
    x = Dense(nb_classes,activation='softmax')

    return x
