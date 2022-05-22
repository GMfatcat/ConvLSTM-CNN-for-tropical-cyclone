# import library

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model

# Preprocess
data = np.load('./images.npy') # change to your .npy path
print('Original size:',data.shape)

# Drop last three for convinence
new_data_train = data[:-3]
new_data_test = data[:-3]
print('New train & test size:',new_data_train.shape,new_data_test.shape)


# reshape to (batch_size, time_steps, height, width, filters (layers)
new_data_train = np.reshape(new_data_train,(230,5,245,329,1))
new_data_test = np.reshape(new_data_train,(230,5,245,329,1))

new_data_train = new_data_train / 255
new_data_test = new_data_test / 255

print('New train & test size:',new_data_train.shape,new_data_test.shape)

# Define model
from keras.metrics import CosineSimilarity,MeanAbsoluteError,RootMeanSquaredError
seq = Sequential()

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 245, 329, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))

seq.compile(loss='mse', optimizer='adam',metrics=[MeanAbsoluteError(),RootMeanSquaredError()])
seq.summary()

print("start training")
history = seq.fit(new_data_train, new_data_test, batch_size=4, epochs=100, validation_split=0.1)

seq.save("images_seq.h5")
print("model saved")

score = seq.evaluate(new_data_train, new_data_test, verbose=0)
print('Test loss:', score[0],'Test MAE:', score[1],'Test RMSE:', score[2])

# training progress plot
plt.figure(1)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('MAE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("mae.jpg")
plt.show()

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("loss.jpg")
plt.show()

plt.figure(3)
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('RMSE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("rmse.jpg")
plt.show()

# show prediction

seq = load_model("images_seq.h5")
path = './prediction'

# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:
  
  # Create a new directory because it does not exist 
  os.makedirs(path)
  print("The new directory is created!")


# result
which = 100 # record to inspect
track = new_data_train[which][:3, ::, ::, ::]
for j in range(6):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::]) # (1, 3, 245, 329, 1)
    new = new_pos[::, -1, ::, ::, ::] # (1, 245, 329, 1)
    track = np.concatenate((track, new), axis=0) # adds +1 to the first dimension in each loop cycle


track2 = new_data_train[which][::, ::, ::, ::]

for i in range(5):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 3:
        ax.text(1, 3, 'Predictions:'+str(i+1), fontsize=20)#, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory:'+str(i+1), fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.axis("off")
    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth:'+str(i+1), fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 3:
        toplot = new_data_test[which][i - 1, ::, ::, 0]

    plt.axis("off")
    plt.imshow(toplot)
    output_name = os.path.join(path,str(i+1) + '_animate.png')
    plt.savefig(output_name)













