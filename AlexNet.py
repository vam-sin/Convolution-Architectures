from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras

def AlexNet():
    model = Sequential()
    # Conv1
    model.add(Conv2D(96,kernel_size = (11,11),
        strides=(4,4),padding='valid',activation='relu',input_shape=(227,227,3)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid'))
    model.add(BatchNormalization())
    # Conv2
    model.add(ZeroPadding2D(2))
    model.add(Conv2D(256,kernel_size=(5,5),
        strides=1,activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(BatchNormalization())
    # Conv3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(384,kernel_size=(3,3),
        strides=(1,1),activation='relu'))
    # Conv4
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(384,kernel_size=(3,3),
        strides=(1,1),activation='relu'))
    # Conv5
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(256,kernel_size=(3,3),
        strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    # FC6
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    # FC7
    model.add(Dense(4096,activation='relu'))
    # FC8 / Output
    model.add(Dense(1000,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

classifier=AlexNet()
