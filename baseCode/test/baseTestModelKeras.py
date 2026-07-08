import numpy as np
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors


def getMolDescriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    res = []
    for nm, fn in Descriptors._descList:
        res.append(fn(mol))
    return res

def smiles_to_MACCS(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    fp = np.array(fp, float)
    return fp

"""
cation = "C=C/C=C/Cn1cc[n+](C)c1"
anion = "CCCC[C@H](C)[O-]"
crystal = "MCC"
Ts = 80
heating_time = 5
"""


nouveau_cation = "C=C/C=C/Cn1cc[n+](C)c1"
nouvel_anion = "CCCC[C@H](C)[O-]"         
nouvelle_T = 200            
nouveau_temps = 5                
nouveau_type_cellulose = "MCC"      

fp_cat = smiles_to_MACCS(nouveau_cation)
des_cat = getMolDescriptors(nouveau_cation)

fp_ani = smiles_to_MACCS(nouvel_anion)
des_ani = getMolDescriptors(nouvel_anion)

c_avicel = 1 if nouveau_type_cellulose == 'Avicel' else 0
c_mcc = 1 if nouveau_type_cellulose == 'MCC' else 0
c_cellulose = 1 if nouveau_type_cellulose == 'cellulose' else 0

X_new = np.c_[
    [fp_cat], [des_cat], 
    [fp_ani], [des_ani], 
    [nouvelle_T], [nouveau_temps], 
    [c_avicel], [c_mcc], [c_cellulose]
]


model = keras.models.load_model('model_cellulose.keras')
prediction = model.predict(X_new)

print("\n" + "="*40)
print(f"Propriétés de l'essai : T={nouvelle_T}°C, Temps={nouveau_temps}h, Cellulose={nouveau_type_cellulose}")
print(f"Résultat de la prédiction (Solubilité estimée) : {prediction[0][0]:.4f}")
print("="*40)