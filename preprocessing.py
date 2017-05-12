import re
from nltk.corpus import stopwords

def clean_stopwords(dirty_sentence):
    sentence = re.sub('[?]', '', dirty_sentence)
    word_list = sentence.split(' ')
    return [word for word in word_list if word not in stopwords.words('english')]
