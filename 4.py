 
import os 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras import layers, models 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
 
# --- Step 1: Define Paths --- 
base_dir = r"C:\Users\karti\Downloads\CatDogBreedClassifier_runnable\data\train" 
 
# --- Step 2: Define Image Properties --- 
IMG_SIZE = (128, 128) 
BATCH_SIZE = 32 
 
# --- Step 3: Prepare Dataset --- 
datagen = ImageDataGenerator( 
    rescale=1.0/255, 
    validation_split=0.2, 
    shear_range=0.1, 
    zoom_range=0.1, 
    horizontal_flip=True 
) 
 
train_gen = datagen.flow_from_directory( 
 
 
    base_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical', 
    subset='training' 
) 
 
val_gen = datagen.flow_from_directory( 
    base_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical', 
    subset='validation' 
) 
 
# --- Step 4: Build CNN + RNN Model --- 
model = models.Sequential([ 
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)), 
    layers.MaxPooling2D(2,2), 
    layers.Conv2D(64, (3,3), activation='relu'), 
    layers.MaxPooling2D(2,2), 
    layers.Conv2D(128, (3,3), activation='relu'), 
    layers.MaxPooling2D(2,2), 
    layers.Flatten(), 
    # Reshape CNN output to feed into RNN (timesteps, features) 
    layers.Reshape((128, -1)), 
    layers.LSTM(64, return_sequences=False), 
    layers.Dense(64, activation='relu'), 
    layers.Dropout(0.5), 
    layers.Dense(2, activation='softmax') 
]) 
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary() 
 
# --- Step 5: Train Model --- 
history = model.fit(train_gen, validation_data=val_gen, epochs=5) 
 
# --- Step 6: Analyze Model Performance --- 
plt.figure(figsize=(6,4)) 
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.title('RNN-based Cat vs Dog Classification') 
plt.xlabel('Epochs')  
 
 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show() 
 
# --- Step 7: Test with Custom Image --- 
from tensorflow.keras.preprocessing import image 
 
test_img_path = r"C:\Users\karti\Downloads\CatDogBreedClassifier_runnable\data\train\dogs\dog.896.jpg" 
 
img = image.load_img(test_img_path, target_size=IMG_SIZE) 
img_array = image.img_to_array(img) / 255.0 
img_array = np.expand_dims(img_array, axis=0) 
 
pred = model.predict(img_array) 
label_map = list(train_gen.class_indices.keys()) 
predicted_label = label_map[np.argmax(pred)] 
 
print(f"Prediction: {predicted_label}") 
plt.imshow(img) 
plt.title(f"Prediction: {predicted_label}") 
plt.axis('off') 
plt.show() 
 
# --- Step 8: Save the Trained Model --- 
model.save("cat_dog_rnn_model.h5") 
print("Model saved successfully as cat_dog_rnn_model.h5")
