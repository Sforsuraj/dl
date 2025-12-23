# EX. NO: 13  BUILD A DEEP LEARNING MODEL TO GENERATE SMILES IN SMILES DATASET 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Embedding 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
 
# STEP 1: Load or create a small sample SMILES dataset 
 
smiles = ["CCO", "CCC", "CCN", "COO", "CCCO"]  # Example dataset 
 
# STEP 2: Prepare character dictionary 
chars = sorted(list(set(''.join(smiles)))) 
char_to_int = {c: i + 1 for i, c in enumerate(chars)}   # reserve 0 for padding 
int_to_char = {i: c for c, i in char_to_int.items()} 
vocab_size = len(chars) + 1 
 
# STEP 3: Convert SMILES strings to sequences of integers 
seqs = [[char_to_int[c] for c in s] for s in smiles] 
maxlen = max(len(s) for s in seqs) 
padded = pad_sequences(seqs, maxlen=maxlen, padding='pre') 
 
# STEP 4: Split features (X) and labels (y) 
X = padded[:, :-1] 
y = padded[:, -1] 
 
# STEP 5: Define the LSTM model 
model = Sequential([ 
    Embedding(vocab_size, 32),     # input_length is deprecated → removed 
    LSTM(64), 
    Dense(vocab_size, activation='softmax') 
]) 
 
# STEP 6: Compile the model 
 
 
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary() 
 
# STEP 7: Train the model (for demonstration, use few epochs) 
model.fit(X, y, epochs=50, verbose=0) 
# STEP 8: Generate a new SMILES-like string 
seed = [char_to_int[c] for c in "CC"] 
for _ in range(3):  # Generate 3 more characters 
    padded_seed = pad_sequences([seed], maxlen=maxlen-1, padding='pre') 
    pred = np.argmax(model.predict(padded_seed, verbose=0)) 
    seed.append(pred) 
# STEP 9: Convert integers back to characters 
generated = ''.join(int_to_char.get(i, '') for i in seed) 
print("\nGenerated SMILES String:", generated) 
