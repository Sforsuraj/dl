import nltk
from nltk.chat.util import Chat, reflections

nltk.download('punkt')

pairs = [
    ['(hi|hello|hey)', ['Hello!', 'Hi there!']],
    ['how are you', ['I am fine, thanks!']],
    ['(.*)', ['Sorry, I did not understand.']]
]

chatbot = Chat(pairs, reflections)
chatbot.respond("hello")
