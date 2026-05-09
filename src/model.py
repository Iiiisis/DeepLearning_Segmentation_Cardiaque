"""
model.py
--------
Définition de l'architecture U-Net simplifiée utilisée pour la segmentation
cardiaque sur coupes IRM 2D (axiales).

Architecture :
    Encodeur  : 2 blocs Conv→Conv→MaxPool (16 puis 32 filtres) - Extraction des caractéristiques.
    Bottleneck: 1 bloc Conv→Conv           (64 filtres) - Représentation latente compressée.
    Décodeur  : 2 blocs UpSampling→Skip connection→Conv→Conv - Reconstruction spatiale.

Sortie : 4 classes de probabilités par pixel
    - activation Softmax garantit que la somme des probabilités des 4 classes vaut 1 pour chaque pixel

Dépendances : tensorflow
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import TARGET_SHAPE


def unet_model(input_shape: tuple = (*TARGET_SHAPE, 1)) -> tf.keras.Model:
    """
    Construit et compile le modèle U-Net.

    Paramètres
    ----------
    input_shape : tuple (H, W, C), défaut (512, 512, 1) indiquant hauteur, largeur et canal (niveau de gris).
    
    Retourne
    --------
    tf.keras.Model compilé prêt pour l'entraînement et compilé avec :
        - optimiseur : Adam
        - loss       : binary crossentropy
        - métrique   : accuracy
    """
    
    # Couche d'entrée définissant les dimensions attendues
    inputs = layers.Input(shape=input_shape)

    # --- Encodeur (Descente / Contraction) ----------------------------------
    # Bloc 1 : Extrait les caractéristiques de bas niveau (bords, textures)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1) # Réduit la taille spatiale par 2

    # Bloc 2 : Extrait des caractéristiques plus complexes
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    # --- Bottleneck----------------------------------
    # Point de compression maximal, capture le contexte global de l'image
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)

    # --- Décodeur (Montée / Expansion) ----------------------------------
    # Bloc 3 : Remonte la résolution et fusionne avec les détails de l'encodeur
    u4 = layers.UpSampling2D()(c3)                  # Double la taille spatiale
    u4 = layers.Concatenate()([u4, c2])             # skip connection niveau 2 (récupère les détails spatiaux)
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u4)
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(c4)

    # Bloc 4 : Restaure la résolution d'origine
    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c1])   # skip connection niveau 1
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(c5)

    # --- Tête de classification ----------------------------------
    # 4 filtres (car 4 classes : 0=Fond, 1=VD, 2=Myocarde, 3=VG)
    outputs = layers.Conv2D(4, 1, activation="softmax")(c5)
    
    # Création du Modèle
    model = models.Model(inputs, outputs, name="UNet")
    model.compile(optimizer="adam",                      # Optimiseur classique et performant
                loss="sparse_categorical_crossentropy",  # Adapté pour des labels entiers (0,1,2,3) en multi-classes
                metrics=["accuracy"],                    # Suit le pourcentage de pixels correctement classifiés
                )
    return model