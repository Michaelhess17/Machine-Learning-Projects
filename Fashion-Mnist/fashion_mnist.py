import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras
from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
import pandas as pd
from keras.layers.advanced_activations import LeakyReLU

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15, random_state=42)

optimizer = Adam(lr=0.001, epsilon=1e-07, decay=0.0)

def vgg_face(weights_path=None):
    img = Input(shape=(28, 28, 1))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, (3, 3), name='conv1_1')(pad1_1)
    leak1_1 = LeakyReLU()(conv1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(leak1_1)
    conv1_2 = Convolution2D(64, (3, 3), activation='relu', name='conv1_2')(pad1_2)
    drop1_1 = Dropout(0.5)(conv1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(drop1_1)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, (3, 3), name='conv2_1')(pad2_1)
    leak2_1 = LeakyReLU()(conv2_1)
    pad2_2 = ZeroPadding2D((1, 1))(leak2_1)
    conv2_2 = Convolution2D(256, (3, 3), activation='relu', name='conv2_2')(pad2_2)
    drop2_1 = Dropout(0.5)(conv2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(drop2_1)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(128, (3, 3), name='conv3_1')(pad3_1)
    leak3_1 = LeakyReLU()(conv3_1)
    pad3_2 = ZeroPadding2D((1, 1))(leak3_1)
    conv3_2 = Convolution2D(64, (3, 3), activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(32, (3, 3), activation='relu', name='conv3_3')(pad3_3)
    drop3_1 = Dropout(0.5)(conv3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(drop3_1)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(32, (3, 3), activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(16, (3, 3), activation='relu', name='conv4_2')(pad4_2)
    drop4_1 = Dropout(0.5)(conv4_2)
    pad4_3 = ZeroPadding2D((1, 1))(drop4_1)
    conv4_3 = Convolution2D(10, (3, 3), activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    flat = Flatten()(pool4)
    out = Activation('softmax')(flat)

    model = Model(input=img, output=out)

    if weights_path:
        model.load_weights(weights_path)

    return model


model = vgg_face()
model.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 64

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs=epochs, validation_data = (X_val,Y_val),
                              verbose=2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis=1)

results = pd.Series(results,name="class")
submission = pd.concat([pd.Series(range(0,10000),name = "index"),results],axis = 1)

submission.to_csv("cnn_fashion_mnist2.csv",index=False)