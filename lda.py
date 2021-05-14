import pandas as pd
import os
from gensim import corpora, models
import gensim
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
from stop_words import get_stop_words
from wordcloud import WordCloud, STOPWORDS
from glob import glob
from pathlib import Path
import re

#nltk.download('punkt') 

def print_topics(ldamodel, num_topics, num_words):
    topics = ldamodel.print_topics(num_topics=num_topics, num_words = num_words)
    for i, topic in enumerate(topics):
        print("Topic", i)
        words = topic[1].split(" + ")
        for j, word in enumerate(words):
            print("\tWord {}:".format(j+1), word)


def LDA_analysis(church_type = "white_wealthy", num_topics=2, passes=20, num_words=15,
                 stopwords = STOPWORDS, stemmer = PorterStemmer(), tokenizer = TweetTokenizer()):
    
    # Loading text
    f_paths = [y for x in os.walk("data/transcriptions") for y in glob(os.path.join(x[0], '*.txt'))]
    doc_set = []
    for path in f_paths:
        if os.path.basename(Path(path).parent.parent) != church_type:
            continue    

        with open(path, 'r') as f:
            text = f.read()
            doc_set.append(text)
    
    # Processing and tokenization
    texts = []
    for i in doc_set:
        raw = i.lower()
        raw = re.sub('[,\.!?]', '', raw)
        tokens = tokenizer.tokenize(raw)
        tokens = [i for i in tokens if not i in stopwords]
        tokens = [stemmer.stem(i) for i in tokens]
        texts.append(tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Generating and fitting model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=passes)

    print("\nChurch type: {}".format(church_type))
    print_topics(ldamodel, num_topics, num_words)
    
if __name__ == '__main__':

    # Regex Tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizer = TweetTokenizer()

    # Stop words
    en_stop = get_stop_words('en')
    en_stop = STOPWORDS

    # Stemmer
    p_stemmer = PorterStemmer()

    # LDA parameters
    num_topics = 2
    passes = 25

    LDA_analysis("black", tokenizer=tokenizer, stopwords=en_stop, stemmer=p_stemmer, passes=passes, num_topics=num_topics)
    LDA_analysis("white_wealthy", tokenizer=tokenizer, stopwords=en_stop, stemmer=p_stemmer, passes=passes, num_topics=num_topics)
    LDA_analysis("white_rural", tokenizer=tokenizer, stopwords=en_stop, stemmer=p_stemmer, passes=passes, num_topics=num_topics)