# MemoireAntoineISAAC
Code du mémoire - Antoine ISAAC

## Description
Ce dépôt contient le code associé à mon mémoire de Master en Sciences Informatiques intitulé "Approches semi-supervisées et deep learning pour la détection de fractures osseuses en imagerie médicale".

## Structure du Projet
- `/scripts` : Contient les scripts Python utilisés pour implémenter les différentes techniques de machine learning décrites dans le mémoire.
- `/Data` : Contient les jeux de données annotés et non annotés.




### Description des Scripts
- `pseudoLabelXception.py` : Implémente le pseudo-labeling avec le modèle Xception.
- `pseudoLabelInceptionResNetV2.py` : Implémente le pseudo-labeling avec le modèle InceptionResNetV2.
- `ensembleLearning.py` : Utilise des techniques d'ensemble learning avec 6 modèles.
- `autoencoder.py` : Teste sur l'utilisation d'autoencodeurs.
- `optimisationLR.py` : Script pour optimiser le taux d'apprentissage.
- `optimisationPoids.py` : Script pour optimiser les poids des classes.


### Description des données
- `labeledData` : Contient les données annotées, divisées en deux sous-dossiers :
  - `fractured` : Ce dossier contient des images radiographiques où une fracture est présente.
  - `non_fractured` : Ce dossier contient des images radiographiques sans fracture.
- `unlabeledData` : Contient des données radiographiques qui n'ont pas été annotées. Ces données sont utilisées pour le pseudo-labeling

### Exemple de résultats
![Capture d'écran 2024-07-30 094531](https://github.com/user-attachments/assets/c302c8cb-22b6-433e-88f2-5041b3223074)

![1eé](https://github.com/user-attachments/assets/3384f349-795c-4112-8328-d51c3bcef584)


## Contributeurs
- Antoine ISAAC

## Remerciements
Je souhaite exprimer ma profonde gratitude à toutes les personnes qui ont contribué à la réalisation de ce mémoire, notamment mon directeur de mémoire, Monsieur Souhaib Ben Taieb, et Monsieur Victor Dheur.
