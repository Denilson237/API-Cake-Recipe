from flask import Flask,request
import joblib
import os
from collections import OrderedDict
from datetime import datetime
from Recettes_Utilitaires import *

# GESTION DE LA SAUVEGARDE ET DE LA MISE A JOUR DES NOUVELLES DONNEES
filename = "Ressources/Recherches Utilisateurs.csv"

if os.path.exists(filename):
    New_dataRV = pd.read_csv(filename)
    New_dataR = New_dataRV["Contenu Du Frigo"].tolist()
else:
    New_dataR = []

New_data = []

# API
app = Flask(__name__)

# Model
Model = joblib.load('Ressources/Recettes_search.joblib')


# POST
@app.route('/search', methods=['POST'])
def perform_search():
    data = request.get_json()
    search = data.get('query')
    name = data.get("name")

    New_data.append(search)
    New_dataR.extend(New_data)
    df_col = pd.DataFrame({"Nom" : name, 'Contenu Du Frigo' : list(OrderedDict.fromkeys(New_dataR)), "Heure" : datetime.now()})
    df_col.to_csv(filename, index=False)
    
    results = Model.search(search)
    return results


# GET
@app.route('/', methods=['GET'])
def recettes():
    query_results = Model.search('')
    return query_results


