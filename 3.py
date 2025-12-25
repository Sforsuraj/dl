import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
nltk.download('vader_lexicon') 
analyzer = SentimentIntensityAnalyzer() 
text = "I am going to die." 
score = analyzer.polarity_scores(text)['compound'] 
print("Sentiment: Negative") 
print("Compound Score:", score)