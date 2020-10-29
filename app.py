import nltk 
from flask import Flask, render_template, request
import pandas as pd

import re
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity

app = Flask(__name__)

df=pd.read_excel('./Data/dialog_talk_agent.xlsx')
df.ffill(axis = 0,inplace=True)

def text_normalization(text):
    text=str(text).lower() # text to lower case
    spl_char_text=re.sub(r'[^ a-z]','',text) # removing special characters
    tokens=nltk.word_tokenize(spl_char_text) # word tokenizing
    lema=wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list=pos_tag(tokens,tagset=None) # parts of speech
    lema_words=[]   # empty list 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'): 
            pos_val='v'
        elif pos_token.startswith('J'): 
            pos_val='a'
        elif pos_token.startswith('R'): 
            pos_val='r'
        else:
            pos_val='n' 
        lema_token=lema.lemmatize(token,pos_val) 
        lema_words.append(lema_token) 
    
    return " ".join(lema_words)

df['lemmatized_text']=df['Contexto'].apply(text_normalization)
tfidf = TfidfVectorizer() 
x_tfidf = tfidf.fit_transform(df['lemmatized_text']).toarray()
df_tfidf = pd.DataFrame(x_tfidf,columns=tfidf.get_feature_names()) 

def chat_tfidf(text):
    lemma = text_normalization(text) # calling the function to perform text normalization
    tf = tfidf.transform([lemma]).toarray() # applying tf-idf
    cos = 1-pairwise_distances(df_tfidf, tf, metric='cosine') # applying cosine similarity
    index_value = cos.argmax() # getting index value 
    return df['Respuesta'].loc[index_value]

@app.route("/")
def index():    
    return render_template("index.html") 

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    return str(chat_tfidf(userText))

if __name__ == "__main__":    
    app.run()