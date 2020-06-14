#%%
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

DATADIR = "money"

CATEGORIES = ["100", "500", "1000"]

IMG_SIZE = 128

#%% [markdown]
# # data argumentation
#%%

data = []
imgGen = ImageDataGenerator(rotation_range = 90)

def img_rotation():
    for category in CATEGORIES: 

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in os.listdir(path):  # iterate over each image
            try:
                img_array = load_img(os.path.join(path,img))
                img_array = img_to_array(img_array)
                img_array = img_array.reshape((1,) + img_array.shape)

                # rotation
                dir_name = os.path.join("transformations", category)
                i = 1
                for batch in imgGen.flow(img_array, batch_size=1, save_to_dir=dir_name, save_format='jpg', save_prefix='trsf'):
                    i += 1
                    if i > 3:
                        break
            except Exception as e:  # in the interest in keeping the output clean...
                pass
        

img_rotation()


# %%
