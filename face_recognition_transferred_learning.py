#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import VGG16


# In[2]:


img_rows = 224
img_cols = 224 #vgg bts user 224*224 image 

#loading dataset
model = VGG16(weights = 'imagenet', include_top=False, input_shape = ( img_rows, img_cols, 3))


# In[3]:


#displaying total layers
for ( i,layer) in enumerate(model.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[4]:


#creting layer 
def modeladd(bottom_model, num_classes, D=512):
  top_model = bottom_model.output
  top_model = Flatten(name = "flatten")(top_model)
  top_model = Dense(D, activation = "relu")(top_model)
  top_model = Dense(1024,activation='relu')(top_model)
  top_model = Dense(1024,activation='relu')(top_model)
  top_model = Dropout(0.3)(top_model)
  top_model = Dense(num_classes, activation = "softmax")(top_model)

  return top_model


# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2

FC_Head = modeladd(model, num_classes)

newmodel = Model(inputs=model.input, outputs=FC_Head)

print(newmodel.summary())


# In[6]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'faces/train/'
test_data_dir = 'faces/test/'

#data augmentation
train_genimg = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip=True,
    fill_mode = 'nearest')

test_genimg = ImageDataGenerator(rescale=1./255)

#changing batch size
train_batch = 16
test_batch = 10

train_generate = train_genimg.flow_from_directory(
 train_data_dir,
 target_size = (img_rows, img_cols),
 batch_size = train_batch,
 class_mode = 'categorical')

test_generate = test_genimg.flow_from_directory(
 test_data_dir,
target_size = (img_rows, img_cols),
batch_size = test_batch,
class_mode = 'categorical',
shuffle = False)


# In[7]:


from keras.optimizers import RMSprop
from keras.callbacks import  ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("face_VGG.h5",
                                   monitor = "val_loss",
                                   mode="min",
                                   save_best_only = True,
                                   verbose = 1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                         min_delta = 0,
                         patience = 3 ,
                         verbose = 2,
                         restore_best_weights = True)

callbacks = [earlystop, checkpoint]

newmodel.compile(loss = 'categorical_crossentropy',
                optimizer = RMSprop(lr=0.001),
                metrics = ['accuracy'])

nb_train_samples = 30
nb_test_samples = 10
epochs = 3
batch_size = 16

history = newmodel.fit_generator(
train_generate,
steps_per_epoch = nb_train_samples // batch_size,
epochs=epochs,
callbacks=callbacks,
validation_data =test_generate,
validation_steps = nb_test_samples // batch_size)


# In[8]:


from keras.models import load_model

classifier = load_model('face_VGG.h5')


# In[ ]:


import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_dic = { "[0]": "virat" ,
            "[1]": "rohit"
    
}

face_dic_n = { "virat": "virat face",
             "rohit": "rohit face"}

def draw_test(name, pred, im):
    human = face_dic[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, human, (20,60), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)
    
def getRandomImage(path):
    folders = list(filter(lambda x:os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("class -" + face_dic_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+ "/" +image_name)

for i in range(0,2):
    input_im = getRandomImage("faces/test/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()



# In[ ]:




