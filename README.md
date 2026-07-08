# DESDissolutionCellulose
Générateur de DES pouvant dissoudre la cellulose.  
Exploitation de l'article :   
https://link.springer.com/article/10.1186/s13321-025-01018-z
https://github.com/Mengyang2024/ML4IL
https://github.com/Mengyang2024/IonGen

## Architecture

On retrouve plusieurs dossiers:
* **baseCode :** Contient actuellement 2 fichiers important
    * **trainKeras.py :** Entraine le modèle Keras (modèle dans **models/**)
    * **testKeras.py :** Predit avec le modèle et avec plusieurs combinaisons les meilleurs résultats 
* **csv :** Fichiers de base pour l'entrainement des modèles
* **models :** Modèles à utiliser pour faire des prédictions
* **results :** Dossier où les résultats des test sont envoyés (format csv)
* **testOtherKeras :** Test pour essayer de comprendre les erreurs ou améliorer le modèle présenté par l'article.


<br>


## Installation 
```
conda create -n ML4IL python=3.8
conda activate ML4IL
conda install cudatoolkit=11.8.0
conda install cudnn=8.9.2.26 -c anaconda
pip install tensorflow==2.13.0 rdkit==2023.9.4 pandas==1.5.3 scikit-learn==1.3.0 shap==0.44.1 matplotlib

Pour entrainer le modèle : "python.exe trainKeras.py"
Pour prédire avec le modèle : "python.exe testKeras.py"
```
