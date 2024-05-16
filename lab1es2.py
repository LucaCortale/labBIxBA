import os
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

#risose necessarie per utilizzare la tokenizzazione 
nltk.download('punkt')
nltk.download('stopwords')

folder_path = "C:\\Users\\LUCA\\Desktop\\BIxBA\\Lab1Materiale\\wikipedia"

preprocessed_texts = []
file_names = []

#La stemming è il processo di riduzione delle parole alle loro forme di base o radici
stememr = SnowballStemmer("italian")

file_italian_stopwords = open("C:\\Users\\LUCA\\Desktop\\BIxBA\\Lab1Materiale\\stopwordsEnglish.txt","r")

italian_stopwords = set(file_italian_stopwords.read().splitlines())

#Loop che legge tutti i testi nella cartella
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path,"r",encoding='utf-8') as file:
        #leggo il testo dal file
        text = file.read()
        #tokenizzo
        tokens = word_tokenize(text)
        #trasformo lowercase
        tokens_lower = [token.lower() for token in tokens]
        #rimuovo le stopword
        tokens_filtered = [token for token in tokens_lower if token not in italian_stopwords]
        #stemming
        tokens_stemmed = [stememr.stem(token) for token in tokens_filtered]
        #Join the tokens back to form preprocessed text
        preprocessed_text = ' '.join(tokens_stemmed)
        #appendo il preprocessamneto alla lista
        preprocessed_texts.append(preprocessed_text)
        #appendo il nome del fule alla lista
        file_names.append(file_name)

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the preprocessed text data
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_texts)
# Convert the TF-IDF matrix to a DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=file_names)
print(tfidf_df)

