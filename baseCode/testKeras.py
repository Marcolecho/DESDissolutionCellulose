import csv
import numpy as np
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd

"""
Chargement du modèle et des combinaisons de tests
"""
LISTE_TEMP_CIBLES = [40, 80, 120, 160, 200]  
LISTE_TEMPS_CIBLES = [0.5, 2.0, 5.0, 10.0]    
LISTE_CELLULOSE_CRYSTALS = ['Avicel', 'MCC', 'cellulose']
model = keras.models.load_model('../models/model_cellulose.keras')

""" 
Pour un SMILES donné, retourne une liste de descripteurs de ce dernier sous forme de tableau
"""
def getMolDescriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: 
        return [0.0] * len(Descriptors._descList)
    return [fn(mol) for nm, fn in Descriptors._descList]

""" 
Pour un SMILES donné, retourne un code unique sous forme de tableau avec des valeurs entre 0 et 1
"""
def smiles_to_MACCS(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: 
        return np.zeros(167, float) 
    return np.array(AllChem.GetMACCSKeysFingerprint(mol), float)


"""
Récupère les maccs et les descripteurs des anions/cations
"""
def calcBestSol(cation, anion, scenarios_physiques, metadonnees):
    fp_cat = smiles_to_MACCS(cation)
    des_cat = getMolDescriptors(cation)
    fp_ani = smiles_to_MACCS(anion)
    des_ani = getMolDescriptors(anion)

    """
    Fusion des descripteurs bruts en vecteur 1D direct
    Assemblage de la partie chimie fixe brute [FP, DESC_RAW]
    Répétition de la chimie pour correspondre au nombre de scénarios physiques
    """
    X_desc_raw = np.concatenate([des_cat, des_ani])
    chimie_fixe = np.concatenate([fp_cat, fp_ani, X_desc_raw])
    chimie_repete = np.tile(chimie_fixe, (len(scenarios_physiques), 1))
    X_giant = np.hstack([chimie_repete, scenarios_physiques])

    """
    Lance la prédiction et récupère la combinaison donnant le meilleur résultat
    Si le résultat est sous 0, on le met à 0, 
    On retourne le résutat 
    """ 
    predictions = model.predict(X_giant, verbose=0) 
    index_meilleur = np.argmax(predictions) 
    meilleure_solubilite = predictions[index_meilleur][0]
    temp_opt, temps_opt, cellulose_opt = metadonnees[index_meilleur]

    if meilleure_solubilite < 0: meilleure_solubilite = 0.0

    return {
        'meilleure_solubilite': meilleure_solubilite,
        'temp_opt': temp_opt,
        'temps_opt': temps_opt,
        'cellulose_opt': cellulose_opt
    }





"""
Génération de la matrice des de paramètres
Pour chaque cristal, de temps et de température de chauffe :
    On stocke toutes les combinaisons
    scenarios_physiques stocke les scénarios au format IA
    metadonnees stocke au format lecture humaine et pour le fichier final
"""
def generationCombaisons(LISTE_CELLULOSE_CRYSTALS, LISTE_TEMP_CIBLES, LISTE_TEMPS_CIBLES):
    scenarios_physiques = []
    metadonnees = [] 
    for cellulose_crystal in LISTE_CELLULOSE_CRYSTALS:
        c_avicel = 1.0 if cellulose_crystal == 'Avicel' else 0.0
        c_mcc = 1.0 if cellulose_crystal == 'MCC' else 0.0
        c_cellulose = 1.0 if cellulose_crystal == 'cellulose' else 0.0
        
        for temp in LISTE_TEMP_CIBLES:
            for heating_time in LISTE_TEMPS_CIBLES:
                scenarios_physiques.append([float(temp), float(heating_time), c_avicel, c_mcc, c_cellulose])
                metadonnees.append((temp, heating_time, cellulose_crystal))

    scenarios_physiques = np.array(scenarios_physiques)
    print(f"Grille prête avec {len(scenarios_physiques)} scénarios")
    return scenarios_physiques, metadonnees


if __name__ == '__main__':
    """ 
    Chargement des fichiers anions/cation et calcul des combinaisons de test
    """
    df_cations = pd.read_csv('../csv/pubchem_cations.csv')
    df_anions = pd.read_csv('../csv/pubchem_anions.csv')
    scenarios_physiques, metadonnees = generationCombaisons(LISTE_CELLULOSE_CRYSTALS, LISTE_TEMP_CIBLES, LISTE_TEMPS_CIBLES)

    fileName = "../results/resultKeras.csv"

    """
    Ouverture et nettoyage du fichier  
    """
    with open(fileName, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Cation', 'Anion', 'Meilleure Solubilite', 'Temp Opt', 'Temps Opt', 'Cellulose Opt'])

    """
    Pour chaque combinaison de anions/cations, lance le calcul de la meilleure prédiction avec toutes les combibaisons différentes
        Retourne la meilleure combinaison avec le meilleure résultat
        Si le résultat est supérieur à 0, on écrit la ligne dans le fichier
    """
    for cation in df_cations['smiles'].values:
        for anion in df_anions['smiles'].values:
            print(f"Calcul en cours -> Cation: {cation} | Anion: {anion}")
            
            bestScore = calcBestSol(cation, anion, scenarios_physiques, metadonnees)

            if bestScore['meilleure_solubilite'] > 0:
                with open(fileName, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        cation, 
                        anion, 
                        f"{bestScore['meilleure_solubilite']:.4f}",
                        bestScore['temp_opt'],
                        bestScore['temps_opt'],
                        bestScore['cellulose_opt']
                    ])