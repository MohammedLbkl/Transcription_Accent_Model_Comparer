# Classification des accents & Évaluation de transcription audio

Ce projet a pour objectif d’évaluer plusieurs modèles de transcription automatique de la parole, puis de classifier des accents.

## Objectifs

- **Tâche 1** : Évaluation des modèles de transcription audio à l’aide du **Word Error Rate (WER)** et mesure des temps de transcription
- **Tâche 2** : Classification automatique des accents à partir des fichiers audio et des transcriptions
- **Données** : Corpus Common Voice (Mozilla) — Enregistrements audio, accents, phrases, durées, etc.

# Installation

Cloner le dépôt :  
```bash
git clone https://github.com/ton-utilisateur/accent-classification-asr.git
cd accent-classification-asr
```
### Installer les dépendances

```bash
pip install -r requirements.txt
```

# Description des données

Les données proviennent du corpus Common Voice de Mozilla, disponible ici :  
[https://commonvoice.mozilla.org/fr/datasets](https://commonvoice.mozilla.org/fr/datasets).  

Ce dataset contient des fichiers audio au format `.mp3` avec leurs accents associés, la durée, l’âge et le sexe du locuteur, ainsi que les phrases prononcées.  
Pour ce projet, seules les informations concernant l’accent et la phrase sont utilisées.  

Les fichiers audio ont été convertis en `.wav` pour une meilleure qualité sonore, car ce format est non compressé et mieux adapté aux traitements acoustiques et à l’extraction de caractéristiques.

# Évaluation des modèles de transcription

Les modèles testés sont :  
- Whisper : tiny, base, small, medium, turbo, large  
- Vosk  

La métrique principale utilisée est le **Word Error Rate (WER)**

La durée moyenne de transcription a également été calculée pour chaque modèle afin d’évaluer la rapidité.

## Résultats moyens obtenus

| Modèle           | WER moyen (%) | Durée moyenne (secondes) |
|------------------|--------------:|-------------------------:|
| Vosk             |         18.58 |                     1.78 |
| Whisper (tiny)   |         14.85 |                     2.93 |
| Whisper (base)   |         11.53 |                     5.19 |
| Whisper (small)  |          5.88 |                    14.25 |
| Whisper (medium) |          5.97 |                    38.97 |
| Whisper (turbo)  |          2.75 |                    66.93 |
| Whisper (large)  |          4.55 |                    83.00 |


# Classification des accents

## Modèles testés

- Régression Logistique  
- Forêt Aléatoire  
- AdaBoost  
- KNN  
- SVM  
- XGBoost  

## Résultats et comparaison

Le modèle Régression Logistique s’est révélé être le plus performant globalement pour la classification des accents, avec un bon compromis entre précision, rappel et F1-score.

| Modèle                | Précision | Rappel | F1-score |
|-----------------------|-----------|--------|----------|
| Régression Logistique | 0.82      | 0.73   | 0.76     |
| Forêt Aléatoire       | 0.81      | 0.60   | 0.63     |
| AdaBoost              | 0.54      | 0.36   | 0.37     |
| KNN                   | 0.76      | 0.60   | 0.61     |
| SVM                   | 0.79      | 0.60   | 0.61     |
| XGBoost               | 0.93      | 0.69   | 0.72     |

---