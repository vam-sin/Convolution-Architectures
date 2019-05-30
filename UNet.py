from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

def UNet():
    model = Sequential()
    # Set 1
    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(572,572,1)))
    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Set 2
    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Set 3
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Set 4
    model.add(Conv2D(512,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Set 5
    model.add(Conv2D(1024,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(1024,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(UpSampling2D(size=(2,2)))

    # Set 6
    model.add(Conv2D(512,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(UpSampling2D(size=(2,2)))

    # Set 7
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(UpSampling2D(size=(2,2)))

    # Set 8
    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(UpSampling2D(size=(2,2)))

    # Set 9
    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(2,kernel_size=(1,1)))

    model.add(Flatten())

    model.summary()

    return model

classifier = UNet()
