from nltk.corpus import stopwords

def clean_stopwords(sentence):
    word_list = sentence[:]
    return [word for word in word_list if word not in stopwords.words('english')]
