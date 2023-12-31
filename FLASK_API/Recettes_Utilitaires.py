import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import jsonify
import joblib

filename = "Ressources/"

df = pd.read_csv(f"{filename}Recettes_token.csv")

class NLTKSearch:
    def __init__(self,df):
        self.df = df
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df["Ingredient_stem_token"]).toarray()
        self.stop_words = list(set(stopwords.words('french')))
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
    
    def search(self, query):
        
        new_stopwords =["tout","ramolli","rapide","extrait","soude","usage","cuillere",
                "petite","belle","commencer","melange","heure","recette", "surgelé",
                "avant","rouge","sortir","kg","g","si","dénoyaute ",
                "pincée","ça","ou","fondu","mou","soupe","coulis",
                "bombée","pot","litre","peu","froid","non","fin","c.","gros","sans","grosse","en","a","etc"]

        self.stop_words.extend(new_stopwords)
        
        def Pretraitement(text):
    
            def remove_s(word):
                if word.endswith('s'):
                    return word[:-1]
                else:
                    return word

            #listes = 
            liste1 = ["cc "," oise ","/ ","cac","cl ","ml "]

            # Supprimer des ponctuations
            text = re.sub("[^\w\s]", " ", text)
            
            text = ' '.join([self.ps.stem(unidecode(word.lower())) for word in word_tokenize(text) if word.isalnum() and word.lower() not in self.stop_words])

            # supprimer les chiffres
            text = re.sub("\d", "", text)

            for i in liste1:
                text = text.replace(i," ")

            #text = [remove_s(word) for word in text]

            text = text.replace("ee","e")

            text = list(set(text.split()))

            text = ' '.join([remove_s(word) for word in text])

            return text
        
        processed_query = Pretraitement(query)
        
        
        tif2 = TfidfVectorizer()

        cosine_similarite = []
        
        for i in range(len(df["Ingredient_stem_token"])):
            tfidf2 = tif2.fit_transform([df["Ingredient_stem_token"][i]]).toarray()
            cosine_similaritie = cosine_similarity(tfidf2, tif2.transform([processed_query]))[0]
            cosine_similarite.append(cosine_similaritie)
        
        cosine_similaritie = np.array(cosine_similarite)
        df['Similarite_%'] = 100*cosine_similaritie.flatten()
        df_tri = df.sort_values(by='Similarite_%', ascending=False)
        df_tri_json = df_tri.to_json(orient='records')
        return jsonify({'tri_json': df_tri_json})

# Créer une instance du moteur de recherche
search_ = NLTKSearch(df)

# Sauvegarder le moteur de recherche avec joblib
joblib.dump(search_, f"{filename}Recettes_search.joblib")