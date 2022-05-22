# view_images.py
# Written By Connor Cozad
# Modified by GMfatcat
#
# Purpose of this file:
# This file is not a necessary part of preparing data or running the neural network. It gives the user a glimpse of
# what the images of hurricanes look like. These are the images that the neural network is trained on in model.py
#
# Outline of this file:
# - Reads numpy files containing hurricane images and the wind speed they are associated with
# - Shows N random satellite images and their wind speeds, N = show_img
# - When running the script, closing the current matplotlib window will cause the next one to open

# Define
show_img = 5
save_fig = False

import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import load_model

def standardize_data(train_images):
    train_images[train_images < 0] = 0
    st_dev = np.std(train_images)
    mean = np.mean(train_images)
    train_images = np.divide(np.subtract(train_images, mean), st_dev)
    return train_images



images = np.load('images.npy')
labels = np.load('labels.npy')

if show_img > images.shape[0]:
    raise ValueError("Exceed image numbers!!")



model = load_model('./result3/5_fold_model.h5')  # load your .h5 file
print("Load modelsuccess...")

# Predict
pred = model.predict(standardize_data(images))
flat_pred = pred.flatten()


for x in range(show_img):
    i = random.randint(0, images.shape[0])
    image = np.reshape(images[i], (images[i].shape[0], images[i].shape[1]))
    plt.figure(x+1)
    plt.imshow(image, cmap='binary')
    title = 'Image #' + str(i) + '   ' + str(labels[i]) + ' knots pred:' + str(round(flat_pred[i],2)) + ' knots'
    plt.title(title)

    if save_fig:
        plt.savefig('Image' + str(i) + '_pred_knots_.png')
  
    plt.show()
