from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

def vgg16():
    model = Sequential()
    model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu',input_shape=(224,224,3)))
    model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu',input_shape=(224,224,3)))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(128,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
    model.add(Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
    model.add(Conv2D(512,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
    model.add(Conv2D(512,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),strides=(1,1),padding='same',
    activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))

    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(4096,activation='relu'))
    model.add(Dense(1000,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    return model

classifier = vgg16()
