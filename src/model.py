"""
model.py
--------
Définition de l'architecture U-Net simplifiée utilisée pour la segmentation
cardiaque sur coupes IRM 2D (axiales).

Architecture :
    Encodeur  : 2 blocs Conv→Conv→MaxPool (16 puis 32 filtres)
    Bottleneck: 1 bloc Conv→Conv           (64 filtres)
    Décodeur  : 2 blocs UpSampling→Skip connection→Conv→Conv

Sortie : carte de probabilités [0,1] par pixel (sigmoid, 1 canal).

Dépendances : tensorflow ≥ 2.x (testé Python 3.7)
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import TARGET_SHAPE


def build_unet(input_shape: tuple = (*TARGET_SHAPE, 1)) -> tf.keras.Model:
    """
    Construit et compile le modèle U-Net.

    Paramètres
    ----------
    input_shape : tuple (H, W, C), défaut (512, 512, 1)

    Retourne
    --------
    tf.keras.Model compilé avec :
        - optimiseur : Adam
        - loss       : binary crossentropy
        - métrique   : accuracy
    """
    inputs = layers.Input(shape=input_shape)

    # ── Encodeur ─────────────────────────────────────────────────────────────
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)

    # ── Décodeur ─────────────────────────────────────────────────────────────
    u4 = layers.UpSampling2D()(c3)
    u4 = layers.Concatenate()([u4, c2])   # skip connection niveau 2
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(u4)
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(c4)

    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c1])   # skip connection niveau 1
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(u5)
    c5 = layers.Conv2D(16, 3, activation="relu", padding="same")(c5)

    # ── Tête de classification ────────────────────────────────────────────────
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    model = models.Model(inputs, outputs, name="UNet")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model