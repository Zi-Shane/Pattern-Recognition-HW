#%% [markdown]
# # PR HW3

#%%
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

DATADIR_1 = "money"
DATADIR_2 = "transformations"

CATEGORIES = ["100", "500", "1000"]

IMG_SIZE = 128


#%% [markdown]
# # 1. read data

#%%
data = []

def create_data(DATADIR):
    for category in CATEGORIES: 

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in os.listdir(path):  # iterate over each image
            try:
                img_array = load_img(os.path.join(path,img))
                img_array = img_to_array(img_array)
                img_array = img_array.reshape((1,) + img_array.shape)

                data.append([img_array, class_num])

            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_data(DATADIR_1)
create_data(DATADIR_2)

print("total data: ", len(data))
# print(data[0])


#%% [markdown]
# # 2. shuffle & split train and test
#%%
import random

random.shuffle(data)

train_size = int(len(data)*0.8)

train = data[:train_size]
test = data[train_size:] 


#%%
# print(len(test))
# for sample in test[:30]:
#     print(sample[1])
# print(train[0][0].shape)
# print(train[15][1])

#%%
X_train = []
y_train = []

for features,label in train:
    X_train.append(features/255.)
    y_train.append(label)

X_test = []
y_test = []

for features,label in test:
    X_test.append(features/255.)
    y_test.append(label)



# X_train = np.array(X_train).reshape(IMG_SIZE, IMG_SIZE, 3)
X_train = np.array(X_train)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_test = np.array(X_test)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# y_train = np.array(y_train)
# y_test = np.array(y_test)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#%%
# print(y_train.shape)

#%%
# print(X_train[0])


#%% [markdown]
# # 3. CNN model

#%%
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

model.add(Conv2D(filters=32, 
                 kernel_size=(3, 3),
                 strides=(1, 1), 
                 padding="same", 
                 input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2 ,2),
                       strides=2))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),
                       strides=2))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation="relu"))

model.add(Dropout(0.25))  # dropout

model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()

# Hyper Parameters
epochs = 10
batch_size = 2
lr = 0.0001
decay = 1e-6
# optimizer = tf.optimizers.RMSprop(lr=lr, decay=decay)
# model.compile(optimizer=optimizer, 
#               loss="categorical_crossentropy", 
#               metrics=["accuracy"])
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs, 
    # validation_data=(X_test, y_test),
)

y_pred = model.predict(X_test)
y_pred_label = np.argmax(y_pred, axis=1)
y_test_label = np.argmax(y_test, axis=1)


# %%
count = 0
for label, predict in zip(y_test_label, y_pred_label):
    if label == predict:
        count += 1
        
print("accruacy: ", count/y_test_label.shape[0])

# %%
