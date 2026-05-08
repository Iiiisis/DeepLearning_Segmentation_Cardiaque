# Segmentation d'Images Médicales Cardiaques (U-Net)

Ce projet implémente un pipeline de Deep Learning pour la segmentation automatique du cœur (ventricule gauche/myocarde) à partir d'images IRM au format NIfTI. Il utilise une architecture **U-Net** simplifiée.

## 📂 Structure du Projet

- `config.py` : Centralisation des chemins de dossiers et des paramètres (taille d'image, epochs).
- `preprocess.py` : Script d'extraction des volumes 3D (.nii) en coupes 2D (.png) avec padding (512x512).
- `train_and_predict.py` : Définition du modèle U-Net, entraînement sur les PNG et génération des prédictions.
- `requirements.txt` : Liste des bibliothèques Python nécessaires.
- `preprocessing.log` : Journal des erreurs et du suivi de l'extraction.

## 🛠 Installation

1. **Créer l'environnement virtuel :**
   ```bash
   python -m venv venv