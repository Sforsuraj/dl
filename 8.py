from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

text = "Pizza have different"
for _ in range(5):
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs).logits[:, -1, :]
    next_id = torch.argmax(logits).item()
    text += " " + tokenizer.decode([next_id])

print(text)
