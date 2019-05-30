from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import keras

def LeNet():
    model = Sequential()
    # Layer C1
    model.add(Conv2D(6,kernel_size=(5,5),strides=(1,1),padding='valid',
                     activation='relu',input_shape=(32,32,3)))
    # Layer S2
    model.add(AveragePooling2D(pool_size=2,strides=2,padding='valid'))
    # Layer C3
    model.add(Conv2D(16,kernel_size=(5,5),strides=(1,1),padding='valid',
                     activation='relu'))
    # Layer S4
    model.add(AveragePooling2D(pool_size=2,strides=2,padding='valid'))
    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(84,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    
    return model

classifier = LeNet()
