import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense 
from tensorflow.keras.models import Model 
import matplotlib.pyplot as plt 
 
# Sample Sales Dataset (User bought Product) 
ratings = np.array([ 
    [0, 0], [0, 1],    # User 0 bought items 0 & 1 
    [1, 1], [1, 2],    # User 1 bought items 1 & 2 
    [2, 0], [2, 2],    # User 2 bought items 0 & 2 

 
 
    [3, 1], [3, 2]     # User 3 bought items 1 & 2 
]) 
 
users = 4 
products = 3 
# Inputs 
x = ratings 
y = np.ones(len(ratings))  # Label: purchase = 1 
 
# Neural Collaborative Filtering Model 
user_in = Input(shape=[1]) 
user_emb = Embedding(users, 5)(user_in) 
user_vec = Flatten()(user_emb) 
 
prod_in = Input(shape=[1]) 
prod_emb = Embedding(products, 5)(prod_in) 
prod_vec = Flatten()(prod_emb) 
 
dot_layer = Dot(axes=1)([user_vec, prod_vec]) 
output = Dense(1, activation="sigmoid")(dot_layer) 
 
model = Model([user_in, prod_in], output) 
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 
 
# Training 
 
 
model.fit([x[:,0], x[:,1]], y, epochs=100, verbose=0) 
print("Training Done!") 
 
# Recommendation for a user 
test_user = 0  # Change this to test other users 
product_scores = [] 
 
for p in range(products): 
    score = model.predict([np.array([test_user]), np.array([p])], verbose=0)[0][0] 
    product_scores.append((p, score)) 
 
print(f"\n Recommendation Scores for User {test_user}:") 
for prod, score in product_scores: 
    print(f"Product {prod}: {score:.3f}") 
# Plotting Recommendation Strength 
prods, scores = zip(*product_scores) 
plt.bar(prods, scores) 
plt.title(f"Recommendation Scores for User {test_user}") 
plt.xlabel("Product ID") 
plt.ylabel("Match Score") 
plt.ylim(0, 1) 
plt.show() 
 
print("\n Recommendation System Executed Successfully!") 
