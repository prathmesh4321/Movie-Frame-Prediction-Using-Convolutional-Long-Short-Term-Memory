from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing.image import load_img
import numpy as np
import pylab as plt
from keras import backend as be

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(1,1),
                   input_shape=(None, 80, 80, 3), #Will need to change channels to 3 for real images
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(2,2),
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(1,1),
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(2,2),
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(1,1),
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(2,2),
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(1,1),
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=40, kernel_size=(2,2),
                   padding='same', return_sequences=True,
                   activation='relu'))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=3, kernel_size=(1,1,1),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')     #loss='categorical_crossentropy'  optimizer='adam'
#be.clear_session()



import os 
movies_input = []
movies_input_shifted = []

for subdir,dirs,files in os.walk("D:\Documents\cnn_lstm\CNN-LSTM-master"):
    for dir in dirs:
        movies_input_delayed = []
        movies_input_shifted_delayed = []  
        for files in os.walk("D:\Documents\cnn_lstm\CNN-LSTM-master"+'\\'+ str(dir)):
            for i in range(len(files[2])):
                img_path = "D:\Documents\cnn_lstm\CNN-LSTM-master"+'\\'+ str(dir) + '\\' + str(files[2][i])
                img = load_img(img_path, target_size=(80,80))
                x = img_to_array(img)
                #print(x)
                x = x //180      #change it acoording to the values after converting image to array
               
                movies_input_delayed.append(x)
        movies_input.append(movies_input_delayed[:-1])
        movies_input_shifted.append(movies_input_delayed[1:])
              
print(np.array(movies_input).shape[0])
print(np.array(movies_input).shape[1])
print(np.array(movies_input).shape[2])
print(np.array(movies_input).shape[3])
print(np.array(movies_input).shape[4])

print(np.array(movies_input_shifted).shape[0])
print(np.array(movies_input_shifted).shape[1])
print(np.array(movies_input_shifted).shape[2])
print(np.array(movies_input_shifted).shape[3])
print(np.array(movies_input_shifted).shape[4])

#Train the network
### Was
#noisy_movies, shifted_movies = generate_movies(n_samples=120)
#seq.fit(noisy_movies[:100], shifted_movies[:100], batch_size=1,
#        epochs=10, validation_split=0.05)
### Now with own images is


seq.fit(np.array(movies_input), np.array(movies_input_shifted), batch_size=1,
        epochs=50)

for subdir,dirs,files in os.walk("D:\Documents\lstm\master"):
    for dir in dirs:
       movies = []
       for files in os.walk("D:\Documents\lstm\master"+'\\'+ str(dir)):
           for i in range(len(files[2])):
                img_path = "D:\Documents\lstm\master"+'\\'+ str(dir) + '\\' + str(files[2][i])
                img = load_img(img_path, target_size=(80,80))
                x = img_to_array(img)
                x = x //180
                movies.append(x)



track = np.array(movies)[:15, ::, ::, ::] 

#print("breakkkkkkkkkkkkkkkkkkkkkkkk")

for j in range(30):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])   #adds 1 extra dimension
    #print(len(new_pos))
    new = new_pos[::, -1, ::, ::, ::]                 #input 15 images and make 30 pred and then reverse
    #print("breakkkkkkkkkkkkk")
    
    track = np.concatenate((track, new), axis=0)        #len of track is 30+15=45
    
#t=track[14,::,::,0]
#print(t)
#plt.imshow(t)
   
"""fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
toplot=new_pos[1,::,::,0]
plt.imshow(toplot)"""

# And then compare the predictions
# to the ground truth
track2 = np.array(movies_input)[0][::, ::, ::, ::]
for i in range(29):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 15:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = np.array(movies_input_shifted)[0][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))