import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


#paths to images that will be trained
train_path = "Dataset-covid/Train"
test_path = "Dataset-covid/Test"


# CNN based model in keras tensorflow

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (224,224,3)))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1,activation = "sigmoid"))
          
model.compile(loss = keras.losses.binary_crossentropy, optimizer = "adam", metrics = ['accuracy'])

#model.summary()

#training data

train_gen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_dataset = image.ImageDataGenerator(
    rescale = 1./255
)

trainer = train_gen.flow_from_directory(
    train_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = "binary"
)

test_gen = test_dataset.flow_from_directory(
    test_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = "binary"
)


hist = model.fit(
    x=trainer,
    steps_per_epoch=4,
    epochs=10,
    validation_data=test_gen,
    validation_steps=2
)