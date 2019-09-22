import csv 
import numpy as np
import cv2
from matplotlib.pyplot import imread
from keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Dropout,Lambda,AveragePooling2D,Cropping2D, ZeroPadding2D,Convolution2D,Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.models import Sequential, Model
from keras.losses import mse
from keras.optimizers import Adadelta,adam,Adam
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications import InceptionV3
from keras.regularizers import l2


PATIENCE = 50
MULTI_PROCESSING = False
THREADS = 1
input_shape = (160,320,3)
batch_size = 64
epochs = 4


lines =[]
with open("data1/driving_log.csv") as f:
    reader=csv.reader(f)
    for line in reader:
        lines.append(line)
images = []
measurements =[]

def process_image(image):
    """
    Preprocess input image
    """
    return imread(image)
    
for data in lines:
    path='data1/IMG/'
    center_image = data[0]
    left_image = data[1]
    right_image = data[2]
    center_image_filename = center_image.split('/')[-1]
    left_image_filename = left_image.split('/')[-1]
    right_image_filename = right_image.split('/')[-1]
    center_image_processed = process_image( path + center_image_filename)
    left_image_processed = process_image( path + left_image_filename)
    right_image_processed = process_image( path + right_image_filename)
    
    steering_center = float(data[3])
    correction = 0.2 # this is a parameter to tune
    measurement_left = steering_center + correction
    measurement_right = steering_center - correction
    
    images.extend((center_image_processed,left_image_processed,right_image_processed))
    measurements.extend((steering_center,measurement_left,measurement_right))

print("Total images {0} and measurements {1}".format(len(images),len(measurements)))
x_train = np.array(images)
y_train = np.array(measurements)



#Basic convolution model
def basic_model():
    """
    Basic keras convolution model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1,kernel_initializer='random_uniform', bias_initializer='zeros'))
    return model

    
#VGG 16 model
def VGG_16():
    """
    VGG 16 keras model architecture
    """
    vgg16 = Sequential()
    #normalization and data augmentation
    vgg16.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    vgg16.add(Cropping2D(cropping=((50,20), (0,0))))
    # Layer 1 & 2
    vgg16.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0)))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(64, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 3 & 4
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(128, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(128, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(MaxPooling2D(pool_size=(2, 2)))
    # Layer 5, 6, & 7
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(256, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(256, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(256, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(MaxPooling2D(pool_size=(2, 2)))
    # Layers 8, 9, & 10
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(MaxPooling2D(pool_size=(2, 2)))
    # Layers 11, 12, & 13
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(ZeroPadding2D((1, 1)))
    vgg16.add(Conv2D(512, (3, 3), padding='same'))
    vgg16.add(Activation('relu'))
    vgg16.add(MaxPooling2D(pool_size=(2, 2)))
    # Layers 14, 15, & 16
    vgg16.add(Flatten())
    vgg16.add(Dense(4096))
    vgg16.add(Activation('relu'))
    vgg16.add(Dropout(0.5))
    vgg16.add(Dense(1000))
    vgg16.add(Activation('relu'))
    vgg16.add(Dropout(0.5))
    vgg16.add(Dense(1))
    return vgg16

#Inception Model
def Inception():
    """
    Inception V3 model architecture for transfer learning
    """
    FC_LAYERS = [120, 84]
    dropout = 0.5
    #Inception model
    base_model = InceptionV3(weights="imagenet",include_top=False,input_shape=(160,320,3))
    finetune_model = build_finetune_model(base_model, 
                                        dropout=dropout, 
                                        fc_layers=FC_LAYERS)
    return finetune_model

def build_finetune_model(base_model, dropout, fc_layers):
    """
    Funtion to add layers on top of pretained model
    """
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New prediction layer
    predictions = Dense(1)(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


def Lenet():
    """
    Yan LeCun's LeNet architecture
    """
    #More deep model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=1))
    return model

WEIGHTS_FILE="LeNet.h5"
model = Lenet()
model.load_weights(WEIGHTS_FILE)
model.summary()
adam = Adam(lr=0.00001)
model.compile(adam, loss='mse', metrics=['accuracy'])
filepath="LeNet-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
stop = EarlyStopping(monitor="val_acc", patience=PATIENCE, mode="max")
#reduce_lr = ReduceLROnPlateau(monitor="val_acc", factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")
callbacks_list = [checkpoint,stop]


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks_list,
          verbose=1,
          workers=THREADS,
          use_multiprocessing=MULTI_PROCESSING,
          validation_split=0.2,shuffle=True)
model.save('LeNet.h5')
