from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class CollegeBot:
    def __init__(self):
        self.tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def reply(self, text):
        ids = self.tok.encode(text, return_tensors="pt")
        out = self.model.generate(ids, max_length=80)
        return self.tok.decode(out[0], skip_special_tokens=True)

bot = CollegeBot()
print("College Bot: Type 'quit' to exit")

while True:
    q = input("You: ")
    if q == "quit": break
    print("Bot:", bot.reply(q))
