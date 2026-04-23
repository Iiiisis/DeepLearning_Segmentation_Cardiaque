from tensorflow.keras import layers, models

def unet_model(input_shape=(512, 512, 1)):
    """
    Construit et compile une architecture réseau de type U-Net simplifiée.

    Args:
        input_shape (tuple): Les dimensions de l'image en entrée (hauteur, largeur, canaux).

    Returns:
        tf.keras.Model: Le modèle U-Net compilé prêt pour l'entraînement.
    """
    inputs = layers.Input(shape=input_shape)
    
    # --- ENCODEUR (Contraction) ---
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)
    
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)
    
    # --- BOTTLENECK (Goulot d'étranglement) ---
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    
    # --- DÉCODEUR (Expansion) ---
    u4 = layers.UpSampling2D()(c3)
    u4 = layers.Concatenate()([u4, c2]) # Connexion résiduelle (Skip connection)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)
    
    u5 = layers.UpSampling2D()(c4)
    u5 = layers.Concatenate()([u5, c1]) # Connexion résiduelle (Skip connection)
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(c5)
    
    # --- COUCHE DE SORTIE ---
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5) # Sigmoïde pour une segmentation binaire
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
