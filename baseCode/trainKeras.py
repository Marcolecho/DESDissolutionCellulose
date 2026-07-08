import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

""" 
Pour un SMILES donné, retourne une liste de descripteurs de ce dernier sous forme de tableau
"""
def getMolDescriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [0.0] * len(Descriptors._descList) 
    res = []
    for nm, fn in Descriptors._descList:
        res.append(fn(mol))
    return res

""" 
Pour un SMILES donné, retourne un code unique sous forme de tableau avec des valeurs entre 0 et 1
"""
def smiles_to_MACCS(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(167, float)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    fp = np.array(fp, float)
    return fp

if __name__ == '__main__':

    """
    Chargement du fichier et découpe du fichier en variable
    """
    print("Chargement des données chimiques...")
    df = pd.read_csv('../csv/dataset_for_cellulose_solubility_ML_model_water_content_less_1%.csv')
    
    cation = list(df.loc[:, 'cation'])
    anion = list(df.loc[:, 'anion'])
    Ts = list(df.loc[:, 'T'])
    heating_time = list(df.loc[:,'heating_time'])
    Ys = list(df.loc[:, 'solv'])
    
    """
    OneHotEncoding sur la colonne crystal
    """
    Crystal = list(df.loc[:, 'cellulose_crystal'])
    Crystal_avicel = []
    Crystal_MCC = []
    Crystal_cellulose = []
    for item in Crystal:
        if item == 'Avicel':
            Crystal_avicel.append(1)
            Crystal_MCC.append(0)
            Crystal_cellulose.append(0)
        elif item == 'MCC':
            Crystal_avicel.append(0)
            Crystal_MCC.append(1)
            Crystal_cellulose.append(0)
        elif item == 'cellulose':
            Crystal_avicel.append(0)
            Crystal_MCC.append(0)
            Crystal_cellulose.append(1)

    """
    Calcul des descripteurs et maccs pour anions et cations
    """
    print("Calcul des descripteurs moléculaires avec RDKit...")
    Des_cation = np.array([getMolDescriptors(c) for c in cation])
    Des_anion = np.array([getMolDescriptors(a) for a in anion])
    fp_anion = np.array([smiles_to_MACCS(a) for a in anion])
    fp_cation = np.array([smiles_to_MACCS(c) for c in cation])
    
    
    """
    Assemblage de la matrice globale X (759 variables d'entrée) et Y
    """
    X = np.c_[fp_cation, Des_cation, fp_anion, Des_anion, Ts, heating_time, Crystal_avicel, Crystal_MCC, Crystal_cellulose]
    Y = np.array(Ys).reshape(len(Ys), 1)


    """
    Configuration hardware & graine aléatoire pour retomber sur le même résultat 
    """
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    keras.utils.set_random_seed(1)

    """
    Définition des paramètres de Keras 3
    """
    print("\nInitialisation de l'architecture du réseau de neurones...")
    model = keras.Sequential([
        keras.layers.Input(shape=(759,)),
        keras.layers.Dense(units=759, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=759, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=1)
    ])

    """
    Utilisation d'un optimiseur et de la manière de noter la qualité du modèle
    """
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    """
    Entrainement du modèle sur 100% du dataset 
    """
    model.fit(X, Y, epochs=7000, batch_size=32, verbose=1)

    """
    Sauvegarde du modèle
    """
    print("\nEntraînement achevé.")
    model.save('../models/model_cellulose.keras')
    print("Modèle FINAL de production exporté avec succès dans './models/model_cellulose.keras' !")