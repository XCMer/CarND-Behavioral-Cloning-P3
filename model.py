import csv
import os
import cv2
import sklearn
import numpy as np
import matplotlib.image as mpimg

from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Conv2D, MaxPooling2D, Activation, Dropout
from keras.callbacks import CSVLogger
import keras

# The most important line that gave me a lot of headache. This line
# makes the difference between whether the image is represented as hxw
# instead of wxh.
keras.backend.set_image_dim_ordering('tf')
csv_logger = CSVLogger('training_log.csv', append=True, separator=';')

# Load samples
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    # ignore header
    next(reader)

    for row in reader:
        samples.append(row)

# Split samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Generate samples
def samples_generator(samples, batch_size=32):
    num_samples = len(samples)
    
    # Infinite loop
    while True:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                # Straight image
                images.append(mpimg.imread('./data/IMG/' + batch_sample[0].split('/')[-1]))
                angles.append(float(batch_sample[3]))

                # Left straight image
                images.append(mpimg.imread('./data/IMG/' + batch_sample[1].split('/')[-1]))
                angles.append(float(batch_sample[3]) + 0.2)

                # Right straight image
                images.append(mpimg.imread('./data/IMG/' + batch_sample[2].split('/')[-1]))
                angles.append(float(batch_sample[3]) - 0.2)

                # Reverse image
                images.append(np.fliplr(mpimg.imread('./data/IMG/' + batch_sample[0].split('/')[-1])))
                angles.append(-float(batch_sample[3]))

                # Left reverse image
                images.append(np.fliplr(mpimg.imread('./data/IMG/' + batch_sample[1].split('/')[-1])))
                angles.append(-(float(batch_sample[3]) + 0.2))

                # Right reverse image
                images.append(np.fliplr(mpimg.imread('./data/IMG/' + batch_sample[2].split('/')[-1])))
                angles.append(-(float(batch_sample[3]) - 0.2))

            # These arrays are uint8 by default. If not converted to float, they'll
            # all most likely become 0 during the normalization step.
            X_train = np.array(images).astype(np.float32)
            y_train = np.array(angles).astype(np.float32)

            yield sklearn.utils.shuffle(X_train, y_train)


# Create instances of generators
# We do data augmentation for each row in the CSV, thus yielding 6 images
# per row in the CSV
augmentation_count = 6

# The batch size being passed to the generator. These are the number
# of rows the generator reads from the CSV
generator_batch_size = 512

# For every line, we yield multiple augmented images
batch_size = augmentation_count * generator_batch_size
train_generator = samples_generator(train_samples, batch_size=generator_batch_size)
validation_generator = samples_generator(validation_samples, batch_size=generator_batch_size)


# Helper functions
def normalize_pixels(x):
    return x/127.5 - .1

def conv_layer(model, conv_depth, conv_size, conv_padding, subsample):
    model.add(Conv2D(conv_depth, *conv_size, subsample=subsample, border_mode='valid'))
    model.add(Activation('relu'))


# Model
model = Sequential()

# Crop unneeded parts of the image, esp. from the top and bottom. We'll keep the sides
# untouched
# Output would be (65, 320)
model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=(160, 320, 3)))

# Normalize pixels
model.add(Lambda(normalize_pixels))

# Convolution layers
conv_layer(model, 24, (5,5), 'valid', (2,2))
conv_layer(model, 36, (5,5), 'valid', (2,2))
conv_layer(model, 48, (5,5), 'valid', (2,2))
conv_layer(model, 64, (3,3), 'valid', (1,1))
conv_layer(model, 64, (3,3), 'valid', (1,1))

model.add(Flatten())

model.add(Dense(1164, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*augmentation_count, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*augmentation_count, nb_epoch=5, verbose=1, callbacks=[csv_logger])

model.save('model.h5')
