import tensorflow as tf 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 
import matplotlib.pyplot as plt 
 
# Load MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
 
# Normalize image data 
x_train = x_train / 255.0 
x_test = x_test / 255.0 
 
# Build Model 
model = Sequential([ 
    Flatten(input_shape=(28,28)), 
    Dense(128, activation='relu'), 
    Dense(10, activation='softmax') 
]) 
 
# Compile 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 
 
 
# Train model 
history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test)) 
 
# Evaluate 
test_loss, test_acc = model.evaluate(x_test, y_test) 
print("✅ Test Accuracy:", test_acc) 
 
# Plot Training Accuracy 
plt.plot(history.history['accuracy'], label="Train Accuracy") 
plt.plot(history.history['val_accuracy'], label="Val Accuracy") 
plt.title("Accuracy Curve") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.legend() 
plt.grid() 
plt.show() 
 
# Prediction Display 
pred = model.predict(x_test[:9]) 
plt.figure(figsize=(7,7)) 
for i in range(9): 
    plt.subplot(3,3,i+1) 
    plt.imshow(x_test[i], cmap='gray') 
    plt.title("Pred: " + str(pred[i].argmax())) 
    plt.axis('off') 
plt.suptitle("Model Predictions on Test Images") 
plt.show() 
