"""
Module de prétraitement des données IRM/Scanner.
Extrait les coupes 2D des volumes NIfTI (.nii) et les sauvegarde au format PNG.
"""

import os
import numpy as np
import nibabel as nib
import imageio.v2 as imageio # Utilisation de v2 pour éviter les avertissements de dépréciation
from tqdm import tqdm

def pad_image(image, target_shape=(512, 512)):
    """
    Applique un padding (remplissage avec des zéros) pour centrer l'image 
    dans une matrice de la taille cible.

    Args:
        image (np.ndarray): L'image 2D source à redimensionner.
        target_shape (tuple): La taille finale souhaitée (hauteur, largeur).

    Returns:
        np.ndarray: L'image redimensionnée (padded) avec la forme `target_shape`.
    """
    padded = np.zeros(target_shape, dtype=image.dtype)
    x, y = image.shape
    x_offset = (target_shape[0] - x) // 2
    y_offset = (target_shape[1] - y) // 2
    padded[x_offset:x_offset+x, y_offset:y_offset+y] = image
    return padded

def find_nii_file(folder):
    """
    Cherche le premier fichier portant l'extension .nii dans un dossier donné.

    Args:
        folder (str): Le chemin du dossier à inspecter.

    Returns:
        str ou None: Le chemin complet vers le fichier .nii s'il existe, sinon None.
    """
    if not os.path.isdir(folder):
        return None
    for f in os.listdir(folder):
        if f.endswith(".nii"):
            return os.path.join(folder, f)
    return None

def process_datasets(input_dir, output_dir):
    """
    Parcourt le dossier source des patients, extrait les tranches 2D des volumes .nii,
    leur applique un padding, et les sauvegarde en .png.

    Args:
        input_dir (str): Chemin vers le dossier contenant les données brutes (dossiers patients).
        output_dir (str): Chemin vers le dossier où sauvegarder les images et labels traités.
    """
    image_dir = os.path.join(output_dir, "images")
    label_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    training_path = os.path.join(input_dir, "training")
    patients = os.listdir(training_path)
    print(f"Nombre total de patients trouvés : {len(patients)}")

    for patient in tqdm(patients, desc="Traitement des patients"):
        patient_path = os.path.join(training_path, patient)
        if not os.path.isdir(patient_path):
            continue

        img_folder = os.path.join(patient_path, f"{patient}_frame01.nii")
        lbl_folder = os.path.join(patient_path, f"{patient}_frame01_gt.nii")

        img_file = find_nii_file(img_folder)
        lbl_file = find_nii_file(lbl_folder)

        if not img_file or not lbl_file:
            print(f"[SKIP] Aucun fichier .nii complet trouvé pour {patient}")
            continue

        if os.path.getsize(img_file) == 0 or os.path.getsize(lbl_file) == 0:
            print(f"[SKIP] Fichier vide pour {patient}")
            continue

        try:
            # Chargement des volumes NIfTI
            img_nii = nib.load(img_file)
            lbl_nii = nib.load(lbl_file)
            
            img_data = img_nii.get_fdata()
            lbl_data = lbl_nii.get_fdata()

            if img_data.shape != lbl_data.shape:
                print(f"[WARNING] Dimensions différentes pour {patient} → ignoré.")
                continue

            # Itération sur l'axe Z (profondeur) pour extraire chaque coupe 2D
            for z in range(img_data.shape[2]):
                img_slice = pad_image(img_data[:, :, z])
                lbl_slice = pad_image(lbl_data[:, :, z])
                
                name = f"{patient}_slice{z:03d}.png"
                img_save_path = os.path.join(image_dir, name)
                lbl_save_path = os.path.join(label_dir, name.replace(".png", "_label.png"))
                
                imageio.imwrite(img_save_path, img_slice.astype(np.uint8))
                imageio.imwrite(lbl_save_path, lbl_slice.astype(np.uint8))

        except Exception as e:
            print(f"[ERREUR] Lecture échouée pour {patient} : {e}")
            continue

    print("✅ Extraction terminée.")

# === POINT D'ENTRÉE DU SCRIPT ===
if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\thblt\OneDrive\Bureau\Centrale\Projet Innov\Data\Data"
    OUTPUT_DIR = r"C:\Users\thblt\OneDrive\Bureau\Centrale\Projet Innov\new_dataset"
    process_datasets(INPUT_DIR, OUTPUT_DIR)