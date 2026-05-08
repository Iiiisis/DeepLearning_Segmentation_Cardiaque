import numpy as np
import nibabel as nib
import imageio
from tqdm import tqdm
from config import DATA_DIR, IMAGE_DIR, LABEL_DIR, IMG_SIZE

def pad_image(image, target_shape=IMG_SIZE):
    padded = np.zeros(target_shape, dtype=image.dtype)
    x, y = image.shape
    x_off, y_off = (target_shape[0] - x) // 2, (target_shape[1] - y) // 2
    padded[x_off:x_off+x, y_off:y_off+y] = image
    return padded

def run_preprocessing():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    
    training_path = DATA_DIR / "training"
    patients = [p for p in training_path.iterdir() if p.is_dir()]

    for patient_path in tqdm(patients):
        p_id = patient_path.name
        # Recherche des fichiers .nii
        img_file = next(patient_path.glob(f"{p_id}_frame01.nii"), None)
        lbl_file = next(patient_path.glob(f"{p_id}_frame01_gt.nii"), None)

        if not img_file or not lbl_file:
            continue

        img_data = nib.load(str(img_file)).get_fdata()
        lbl_data = nib.load(str(lbl_file)).get_fdata()

        for z in range(img_data.shape[2]):
            img_slice = pad_image(img_data[:, :, z])
            lbl_slice = pad_image(lbl_data[:, :, z])
            
            name = f"{p_id}_slice{z:03d}.png"
            imageio.imwrite(IMAGE_DIR / name, img_slice.astype(np.uint8))
            imageio.imwrite(LABEL_DIR / name.replace(".png", "_label.png"), lbl_slice.astype(np.uint8))

if __name__ == "__main__":
    run_preprocessing()