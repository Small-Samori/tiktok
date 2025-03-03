import pandas as pd
import re, os, pickle
import gensim
import gensim.corpora as corpora
from tqdm import tqdm
import spacy

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_short_text(data_path):
    data = pd.read_csv(data_path, encoding='utf-8')
    data = data.dropna(subset=['text'])
    data = data.rename(columns={'text': 'message', 'id':'message_id'})
    data = data.reset_index()
    
    threshold = 5
    data['message_split'] = [i.split(' ') for i in data['message']]
    lengths = [len(i) for i in data['message_split']]
    
    drop_ids = [i for i in range(len(lengths)) if lengths[i] < threshold]
    data = data.drop(drop_ids)
    data = data.drop(['message_split', 'index'], axis=1)
    data = data.reset_index().drop(['index'], axis=1) 

    return data

def process_and_tokenize(data_text):
    data_text = [re.sub('\S*@\S*\s?', '', sent) for sent in data_text] # remove emails
    data_text = [re.sub('\+?\d[\d .-]{8,}\d', '', sent) for sent in data_text] #remove phone numbers
    data_text = [re.sub('\s+', ' ', sent) for sent in data_text] # remove newlines
    data_text = [re.sub("\'", "", sent) for sent in data_text] # remove single quotes

    tokenized_data_text = [gensim.utils.simple_preprocess(str(sent), deacc=True, min_len=5, max_len=10**100) for sent in data_text]

    return tokenized_data_text


def ngram_and_lemmatize(tokenized_data_text, make_bigram=True, make_trigram=False, 
                        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    bigram = gensim.models.Phrases(tokenized_data_text, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[tokenized_data_text], threshold=100) 

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    data_text = [[word for word in text if word not in stop_words] for text in tokenized_data_text]
    
    if make_bigram:
        data_text = [bigram_mod[sent] for sent in tqdm(data_text, desc="Making Bigrams ...")]

    if make_bigram and make_trigram:
        data_text = [trigram_mod[bigram_mod[sent]] for sent in tqdm(data_text, desc="Making Trigrams ...")]

    lemm_data_text = []
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in tqdm(data_text, desc="Lemmatizing ..."):
        doc = nlp(" ".join(sent)) 
        lemm_data_text.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
    return lemm_data_text