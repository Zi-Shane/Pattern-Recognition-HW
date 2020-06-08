#%% [markdown]
# # PR HW3
# ## 1. load data

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# from tqdm import tqdm

DATADIR = "money"

CATEGORIES = ["100", "500", "1000"]

IMG_SIZE = 128

# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR,category)  # create path to dogs and cats
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
#         # img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
#         plt.imshow(img_array, cmap='gray')  # graph it
# #         plt.show()  # display!

#         break  # we just want one for now so break
#     break  #...and one more!

# print(img_array.shape)
#%%
data = []

def create_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
                data.append([img_array, class_num])  # add this to our data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_data()

# print(len(data))
# print(data[0])


#%%
import random

random.shuffle(data)

train_size = int(len(data)*0.5)
# test_size = len(data) - train_size

train = data[:train_size]
test = data[train_size:] 


#%%
# print(len(test))
# for sample in test[:60]:
#     print(sample[0][0][0])

#%%
X_train = []
y_train = []

for features,label in train:
    X_train.append(features)
    y_train.append(label)

X_test = []
y_test = []

for features,label in test:
    X_train.append(features)
    y_train.append(label)


# X_train = np.array(X_train).reshape(IMG_SIZE, IMG_SIZE, 3)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#%%
X_train = X_train/255.
print(X_train.shape[1:])
#%%
# print(X_train[0])

#%%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#%%
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

solve_cudnn_error()



#%%
model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same",  input_shape=(3, IMG_SIZE, IMG_SIZE)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation="relu"))

model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()
# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Hyper Parameters
epochs = 30
batch_size = 2
lr = 0.0001
decay = 1e-6
optimizer = tf.optimizers.RMSprop(lr=lr, decay=decay)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'],
#               )

model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs, 
    validation_data=(X_test, y_test),
)

y_pred = model.predict(X_test)


# %%
