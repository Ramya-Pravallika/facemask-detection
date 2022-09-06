# facemask-detection
# importing the libraries and tools
import pandas as pd # data processing
import numpy as np # linear algebra
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Dense,Flatten,BatchNormalization,Conv2D,Dropout
from tensorflow.keras.optimizers import Adam
from keras.layers import MaxPooling2D 
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator # training the data
import matplotlib.pyplot as plt # Image plotting
from google.colab import files # files importing
# uploading the dataset
uploaded=files.upload()
# unzipping the dataset
!unzip data.zip
!rm data.zip
# Training the dataset
batch_size=9
epoch=35
directory='data'
train_datagen= ImageDataGenerator(validation_split=0.2,
                                  rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
train_generator= train_datagen.flow_from_directory(
                                  directory,
                                  target_size = (70, 70),
                                  batch_size = batch_size,
                                  color_mode="rgb",
                                  class_mode ='binary',
                                  seed=2020,
                                  subset = 'training')
validation_generator = train_datagen.flow_from_directory(
                                  directory,
                                  target_size = (70, 70),
                                  batch_size = batch_size,
                                  color_mode="rgb",
                                  class_mode ='binary',
                                  subset = 'validation')
# plotting some random images from the trained dataset
def plotimages(images_arr):
  fig, axes = plt.subplots(1, batch_size, figsize=(20,20))
  axes = axes.flatten()
  for img, ax in zip( images_arr, axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()
plotimages(imgs);
print(labels);
# sequential modelling using convolutional 2D layers and maxpool layers with same padding and activation as relu
model = Sequential([
                    Conv2D(filters=32, kernel_size=(3,3),activation='relu',padding='same' ,input_shape=(70,70,3)),
                    MaxPooling2D(pool_size=(2,2), strides=2),
                    Conv2D(filters=32, kernel_size=(3,3),activation='relu',padding='same'),
                    MaxPooling2D(pool_size=(2,2), strides=2),
                    Conv2D(filters=64, kernel_size=(3,3),activation='relu',padding='same'),
                    MaxPooling2D(pool_size=(2,2), strides=2),
                    Flatten(),
                    Dense(units=64, activation='relu'),
                    Dense(units=1, activation='sigmoid'),
])
model.summary()
# finding accuracy and precesion of the trained model and printing all values of batch size 35
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator ,epochs=epoch,validation_data=validation_generator)
# plotting training and validation loss
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochstoplot = range(1,epoch+1)
plt.plot(epochstoplot, loss_train, 'g' ,label='training loss')
plt.plot(epochstoplot, loss_val, 'b' ,label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
# plotting training and validation accuracy
accur_train = history.history['accuracy']
accur_val = history.history['val_accuracy']
plt.plot(epochstoplot, accur_train, 'g' ,label='training accuracy')
plt.plot(epochstoplot, accur_val, 'b' ,label='validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# finally getting an output image if person is wearing a mask or not
from IPython.display import Image, display
TGREEN = '\033[1;37;42m'
TRED = '\033[1;37;41m'
for i in range (1,15):
  img_directory = str(i) + '.jpg'
  img_pred = image.load_img(img_directory, target_size= (70,70))
  img_pred = image.img_to_array(img_pred)
  img_pred = np.expand_dims(img_pred, axis=0)
  prediction = model.predict(img_pred)
  display(Image(img_directory,width=150, height=150))
  print("\n")
  if(int(prediction[0][0]) ==0):
    print(TGREEN + "The person is wearing a mask. \n")
  else:
    print(TRED + "The person is not wearing a mask. \n")
