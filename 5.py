!pip install translate
from translate import Translator 
translator = Translator(from_lang="en", to_lang="de") 
text = "Hi Germany! Welocme to NLP" 
print(translator.translate(text)) 