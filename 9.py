import numpy as np 
import matplotlib.pyplot as plt 
import networkx as nx 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense 
 
# Load IMDB dataset 
num_words = 5000 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words) 
 
# Pad sequences to equal length 
max_len = 200 
x_train = pad_sequences(x_train, maxlen=max_len) 
x_test = pad_sequences(x_test, maxlen=max_len) 
 
# Create RNN Model 
model = Sequential([ 
    Embedding(num_words, 32, input_length=max_len), 
 
 
    LSTM(64), 
    Dense(1, activation='sigmoid') 
]) 
 
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 
 
# Train (Small epochs for quick output) 
history = model.fit(x_train, y_train, epochs=2, batch_size=128, 
                    validation_data=(x_test, y_test)) 
 
# Plot accuracy graph 
plt.plot(history.history["accuracy"], label="Training Accuracy") 
plt.plot(history.history["val_accuracy"], label="Validation Accuracy") 
plt.title("Sentiment Classification Accuracy") 
plt.legend() 
plt.show() 
print("\n✔ Training Done!") 
 
# Create a small Network Graph for Visualizing Word Relationships 
word_index = imdb.get_word_index() 
index_word = {v+3: k for k, v in word_index.items()} 
 
# Pick one test review to visualize 
sample = x_test[1][:30]  # first 30 words 
words = [index_word.get(i, "?") for i in sample] 
 
# Create Graph 
G = nx.Graph() 
for i in range(len(words) - 1): 
    G.add_edge(words[i], words[i+1]) 
 
plt.figure(figsize=(10,6)) 
nx.draw(G, with_labels=True, node_color="lightblue", font_size=8, edge_color="gray") 
plt.title("Network Graph of Words in Review") 
plt.show() 
 
pred = model.predict(np.array([sample]))[0][0] 
print("\nPredicted Sentiment:", "Positive ✅" if pred > 0.5 else "Negative ❌") 
