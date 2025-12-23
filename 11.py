import tensorflow as tf 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add 
import numpy as np 
# Step 1: Load pre-trained CNN for image feature extraction 
vgg = VGG16(weights='imagenet') 
vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output) 
def extract_features(filename):  
    from tensorflow.keras.preprocessing import image 
    img = image.load_img(filename, target_size=(224,224)) 
    img = image.img_to_array(img) 
    img = np.expand_dims(img, axis=0) 
    img = tf.keras.applications.vgg16.preprocess_input(img) 
    feature = vgg.predict(img, verbose=0) 
    return feature 
# Step 2: Simulated caption data (for demonstration) 
vocab_size = 5000 
max_length = 30 
# Step 3: Define the caption generation model 
inputs1 = Input(shape=(4096,)) 
fe1 = Dropout(0.5)(inputs1) 
fe2 = Dense(256, activation='relu')(fe1) 
inputs2 = Input(shape=(max_length,)) 
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2) 
se2 = LSTM(256)(se1) 
decoder1 = add([fe2, se2]) 
decoder2 = Dense(256, activation='relu')(decoder1) 
outputs = Dense(vocab_size, activation='softmax')(decoder2) 
model = Model(inputs=[inputs1, inputs2], outputs=outputs) 
model.compile(loss='categorical_crossentropy', optimizer='adam') 
print(model.summary()) 
