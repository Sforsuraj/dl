from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def chatbot(text):
    ids = tokenizer.encode(text, return_tensors="pt")
    out = model.generate(ids, max_length=80)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("Hi, I'm your chatbot. Type 'quit' to exit.")
while True:
    msg = input("You: ")
    if msg.lower() == "quit":
        print("Goodbye!")
        break
    print("Bot:", chatbot(msg))
