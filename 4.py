!pip install rake-nltk
import nltk
nltk.download('stopwords') # Download the stopwords corpus
from rake_nltk import Rake
r = Rake()
text = """NLP stands for Natural Language Processing.
It is the branch of Artificial Intelligence that gives the ability to machine understand
and process human languages."""
r.extract_keywords_from_text(text)
print(r.get_ranked_phrases())