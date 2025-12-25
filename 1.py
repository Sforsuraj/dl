from nltk.tokenize import word_tokenize, TweetTokenizer, sent_tokenize 
import nltk; nltk.download('punkt_tab') 
# Word Tokenization 
print("# Before: this is a text ready to tokenize") 
print("# After:", word_tokenize("this is a text ready to tokenize")) 
# Tweet Tokenization 
t=TweetTokenizer() 
print("# Before: This is a tweet @jack #NLP") 
print("# After:", t.tokenize("This is a tweet @jack #NLP")) 
# Sentence Tokenization 
text="This is a sentence. This is another one!\nAnd this is the last one." 
print("# Before: This is a sentence. This is another one!") 
print("# And this is the last one.") 
print("# After:", sent_tokenize(text))