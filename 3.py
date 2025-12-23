#Code 1 — The Model Builder and Trainer


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- Step 1: Path setup ---
data_dir = r"C:\Users\karti\Downloads\CatDogBreedClassifier_runnable\data\train"

# --- Step 2: Train/Validation split automatically ---
img_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% data for validation
)

train_gen = datagen.flow_from_directory(



    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# --- Step 3: Build CNN ---
model = models.Sequential([
    layers.Input(shape=(128,128,3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Step 4: Train ---
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# --- Step 5: Visualize results ---
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Cat vs Dog Classification Accuracy")
plt.show()



#Code 2 — The Image Tester


from tensorflow.keras.preprocessing import image
import numpy as np

# --- Step 6: Test on custom image ---
img_path = r"C:\Users\karti\Downloads\CatDogBreedClassifier_runnable\data\train\dogs\dog.5.jpg"  # change to your test image path

# Load and preprocess image
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(img_array)
label = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

# Show result
print(f"Predicted class: {label}")
plt.imshow(img)
plt.title(f"Prediction: {label}")
plt.axis("off")
plt.show()