import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Reshape, Flatten, Input, LeakyReLU 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.datasets import mnist 
 
# Load MNIST 
(x_train, _), (_, _) = mnist.load_data() 
x_train = (x_train.astype("float32") - 127.5) / 127.5 
x_train = x_train.reshape(-1, 28*28) 
 
# Generator 
generator = Sequential([ 
    Input(shape=(100,)), 
    Dense(256), 
    LeakyReLU(0.2), 
    Dense(28*28, activation="tanh") 
]) 
 
# Discriminator 
discriminator = Sequential([ 
    Input(shape=(28*28,)), 
 
 
    Dense(256), 
    LeakyReLU (0.2), 
    Dense(1, activation="sigmoid") 
]) 
discriminator.compile(optimizer="adam", loss="binary_crossentropy") 
 
# GAN Model 
discriminator.trainable = False 
gan = Sequential([generator, discriminator]) 
gan.compile(optimizer="adam", loss="binary_crossentropy") 
 
# Training 
epochs = 200  # fast training 
batch = 64 
for i in range(epochs): 
    # Real images 
    idx = np.random.randint(0, x_train.shape[0], batch) 
    real = x_train[idx] 
 
    # Fake images 
    noise = np.random.randn(batch, 100) 
    fake = generator.predict(noise, verbose=0) 
 
    # Train discriminator 
    discriminator.trainable = True 
    discriminator.train_on_batch(real, np.ones((batch, 1))) 
    discriminator.train_on_batch(fake, np.zeros((batch, 1))) 
 
    # Train generator 
    discriminator.trainable = False 
    gan.train_on_batch(noise, np.ones((batch, 1))) 
 
    if i % 50 == 0: 
        print(f"Epoch: {i}/{epochs}") 
 
# Generate final fake digits 
noise = np.random.randn(25, 100) 
generated = generator.predict(noise).reshape(-1, 28, 28) 
 
# Display 
plt.figure(figsize=(6, 6)) 
for i in range(25): 
    plt.subplot(5,5,i+1) 
    plt.imshow(generated[i], cmap='gray') 
 
 
    plt.axis('off')  
plt.suptitle("Generated Digits using GAN") 
plt.show() 
 
print("GAN completed successfully!") 
