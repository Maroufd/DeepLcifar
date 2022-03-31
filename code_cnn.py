import numpy as np
import time

import tensorflow as tf
import tensorflow.keras.utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10,cifar100
#for the logs
import datetime

train=[0.8,0.85,0.9]
epoch=[15,20,25,30]
batch=[128,256,512]
for tr in train:
    for ep in epoch:
        for bat in batch:
            log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            # load data set
            (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
            print("x_train.shape=", x_train.shape)
            print("x_train.dtype=", x_train.dtype)
            print("y_train.shape=", y_train.shape)
            print("y_train.dtype=", y_train.dtype)
            print("x_test.shape=", x_test.shape)
            print("x_test.dtype=", x_test.dtype)
            print("y_test.shape=", y_test.shape)
            print("y_test.dtype=", y_test.dtype)

            # Classes
            clases = np.unique(y_train)
            print("clases=", clases)
            num_clases = len(clases)


            # CIFAR 10
            # img_rows = 32
            # img_cols = 32
            input_shape=x_train.shape[1:]

            X_train = x_train
            X_test = x_test

            # Scale input from [0,255] to [0,1]
            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train /= 255
            X_test /= 255
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, train_size=tr, test_size=1-tr, random_state=42)
            print('X_train.shape=', X_train.shape)
            print('Y_train.shape=', Y_train.shape)
            print('X_val.shape=', X_val.shape)
            print('Y_val.shape=', Y_val.shape)


            # Convert labels to one-hot encoding
            Y_train = tensorflow.keras.utils.to_categorical(Y_train, num_clases)
            Y_val = tensorflow.keras.utils.to_categorical(Y_val, num_clases)
            Y_test = tensorflow.keras.utils.to_categorical(y_test, num_clases)

            batch_size = bat
            epochs = ep

            #build NN
            model = Sequential()
            model.add(Conv2D(filters=48, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same'))
            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))


            model.add(Conv2D(filters=192, kernel_size=(3, 3), padding='same'))
            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Conv2D(filters=32, kernel_size=(1, 1), padding='same'))
            model.add(Activation('relu'))

            model.add(Conv2D(filters=10, kernel_size=(4, 4), padding='valid'))
            model.add(Flatten())
            model.add(Activation('softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()


            model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=['accuracy'])
            model.summary()


            start = time.time()

            model.fit(X_train, Y_train, batch_size=batch_size, 
                    epochs=epochs, verbose=1, validation_data=(X_val, Y_val),
                    callbacks=[tensorboard_callback])

            end = time.time()

            score = model.evaluate(X_test, Y_test, verbose=0)
            print('Test loss:', score[0])
            print('Accuracy:', score[1])
            file=open("results_cnn_batch_epoch_train.txt","a")
            print("Training took " + str(end - start) + " seconds")
            file.write("Train size: "+str(tr)+"; Epoch: "+str(ep)+"; Batch: "+str(bat)+"; Accuracy: "+str(score[1])+"; Test loss: "+str(score[0])+"; Training time: "+str(end - start) + " seconds \n")        
            file.close()



