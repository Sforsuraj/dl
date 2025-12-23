import os 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras import layers, models 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
 
base_dir = r"C:/Users/karti/Downloads/CatDogBreedClassifier_runnable/data" 
 
train_dir = os.path.join(base_dir, "train") 
validation_dir = os.path.join(base_dir, "validation") 
 
train_cats_dir = os.path.join(train_dir, 'cats') 
train_dogs_dir = os.path.join(train_dir, 'dogs') 
 
cat_files = os.listdir(train_cats_dir)[:4] 
dog_files = os.listdir(train_dogs_dir)[:4]  
 
 
 
plt.figure(figsize=(8,8)) 
for i, file in enumerate(cat_files + dog_files): 
    img = mpimg.imread(os.path.join( 
        train_cats_dir if i < 4 else train_dogs_dir, file 
    )) 
    plt.subplot(2,4,i+1) 
    plt.imshow(img) 
    plt.axis("off") 
plt.show() 
 
#  Preprocessing 
train_gen = ImageDataGenerator( 
    rescale=1/255, 
    rotation_range=40, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True 
) 
 
val_gen = ImageDataGenerator(rescale=1/255) 
 
train = train_gen.flow_from_directory( 
    train_dir, 
    batch_size=20, 
    target_size=(224,224), 
    class_mode="binary" 
) 
 
val = val_gen.flow_from_directory( 
    validation_dir, 
    batch_size=20, 
    target_size=(224,224), 
    class_mode="binary" 
) 
 
#  Load VGG16 model 
base_model = VGG16( 
    weights="imagenet", 
 
    include_top=False, 
    input_shape=(224,224,3) 
 
 
) 
base_model.trainable = False 
 
model = models.Sequential([ 
    base_model, 
    layers.Flatten(), 
    layers.Dense(512, activation='relu'), 
    layers.Dropout(0.5), 
    layers.Dense(1, activation='sigmoid') 
]) 
 
model.compile( 
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy'] 
) 
 
#  Train the model 
history = model.fit(train, epochs=5, validation_data=val) 
 
# Accuracy Plot 
plt.plot(history.history['accuracy'], label="Train") 
plt.plot(history.history['val_accuracy'], label="Validation") 
plt.title("Model Accuracy") 
plt.legend() 
plt.show()