import spacy 
nlp=spacy.load("en_core_web_sm") 
text="I am a Python developer with 3 years of experience. My expertise includes machine learning, data analysis, and effective communication." 
skills=["python","machine learning","data analysis","communication"] 
doc=nlp(text) 
found=[t.text.lower() for t in doc if t.text.lower() in skills] 
print("Extracted Skills:",found) 
print("Years of Experience:",0) 
print("Meets Screening Criteria:",False) 