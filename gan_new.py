'''
Reference for GAN in ECL
'''
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Initialise the combined network
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_combined(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.add(Flatten(input_shape=self.img_shape, trainable=False))
        model.add(Dense(512, trainable=False))
        model.add(LeakyReLU(alpha=0.2, trainable=False))
        model.add(Dense(256, trainable=False))
        model.add(LeakyReLU(alpha=0.2, trainable=False))
        model.add(Dense(1, activation='sigmoid', trainable=False))

        noise = Input(shape=(self.latent_dim,))
        validity = model(noise)

        return Model(noise, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.full((batch_size, 1), 0.00000001)
        Y_input = np.concatenate((valid, fake))

        # Useful numbers to split com_wts
        gen_layers = len(self.generator.get_weights())
        dis_layers = len(self.discriminator.get_weights())
        com_layers = len(self.combined.get_weights())

        # Get combined weights for initialisation
        com_wts = self.combined.get_weights()

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Generate noise for generated image set
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # To get input layer weights
            wts_fromdisnet = self.discriminator.get_weights()

            # Extract generator and discriminator weights from combined weights
            gen_wts = com_wts[0:gen_layers]
            obtaineddis_wts = com_wts[gen_layers+1:gen_layers + dis_layers]
            dis_wts = []
            dis_wts.append(wts_fromdisnet[0]) #Jugaad because input layer isn't there in combined
            dis_wts.extend(obtaineddis_wts)

            # Generate a batch of new images by setting the new generator weights
            self.generator.set_weights(gen_wts)
            gen_imgs = self.generator.predict(noise)
            
            # Concatenate real and generated images to get X.
            X_input = np.concatenate((imgs, gen_imgs))

            # Set weights which are retrieved from combined weights of previous iteration
            self.discriminator.set_weights(dis_wts)

            # Fit the discriminator
            self.discriminator.fit(X_input, Y_input, batch_size*2, 1)

            # Get the new weights of discriminator to send to combined model
            newdis_wts = self.discriminator.get_weights()
            

            # ---------------------
            #  Train Generator
            # ---------------------

            #Generates noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Combine generator and discriminator weights to get new combined weights. Set the combined weights to the model.
            newcom_wts = []
            newcom_wts.extend(gen_wts)
            newcom_wts.extend(newdis_wts)
            self.combined.set_weights(newcom_wts)
            
            # Fit the combined (to have the discriminator label samples as valid)
            self.combined.fit(noise, valid, batch_size, 1)

            # Get new combined weights
            com_wts = self.combined.get_weights()

            #Set generator weights for obtaining output
            self.generator.set_weights(com_wts[0:gen_layers])

            # Print the epoch number
            print("%d" %epoch)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.canvas.draw()
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=1000, batch_size=100, sample_interval=100)
