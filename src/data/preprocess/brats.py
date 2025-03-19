import os
import random
import nibabel
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

random.seed(54)

def classify_healthy_data(data_dir, src_dir):
    img_types = ['flair', 'seg', 't1', 't1ce', 't2']

    healthy_dir = Path(data_dir) / "healthy"
    unhealthy_dir = Path(data_dir) / "unhealthy"

    # create folder
    healthy_image_dir = Path(healthy_dir) / "image"
    unhealthy_image_dir = Path(unhealthy_dir) / "image"
    unhealthy_mask_dir = Path(unhealthy_dir) / "mask"

    healthy_image_dir.mkdir(parents=True, exist_ok=True)
    unhealthy_image_dir.mkdir(parents=True, exist_ok=True)
    unhealthy_mask_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = Path(src_dir)
    for patient_dir in tqdm(patient_dirs.iterdir(), desc="Patient directories", total=len(list(patient_dirs.iterdir()))):
        if not patient_dir.is_dir(): continue

        patient_healthy_image_des_dir = healthy_image_dir / patient_dir.name
        patient_unhealthy_image_des_dir = unhealthy_image_dir / patient_dir.name
        patient_unhealthy_mask_des_dir = unhealthy_mask_dir / patient_dir.name

        patient_healthy_image_des_dir.mkdir(parents=True, exist_ok=True)
        patient_unhealthy_image_des_dir.mkdir(parents=True, exist_ok=True)
        patient_unhealthy_mask_des_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(patient_dir.iterdir())
        images, mask = [], None

        for file, img_type in zip(files, img_types):
            if img_type not in file.name:
                continue

            img = np.array(nibabel.load(file).get_fdata())

            # crop to a size of (224, 224) from (240, 240)
            img = img[8:-8, 8:-8, ...]

            if img_type == 'seg':
                mask = np.where(img > 0, 1, 0).astype(np.float32)
            else:
                images.append(img)

        image_vol = np.stack(images, axis=2)
        mask_vol = mask
        healthy_count, unhealthy_count = 0, 0
        # exclude the lowest 80 slices and the uppermost 26 slices
        for slice_idx in range(80, 129):
            if mask_vol is not None and mask_vol[..., slice_idx].sum():
                patient_image_des_dir = patient_unhealthy_image_des_dir 
                mask_name = patient_unhealthy_mask_des_dir / f"mask_slice_{slice_idx}.npy"
                np.save(mask_name, mask_vol[..., slice_idx])
                unhealthy_count += 1
            else:
                # no mask
                patient_image_des_dir = patient_healthy_image_des_dir
                healthy_count += 1

            image_name = patient_image_des_dir / f"image_slice_{slice_idx}.npy"
            np.save(image_name, image_vol[..., slice_idx])
        print(f"Patient {patient_dir.name}: {healthy_count} healthy slices, {unhealthy_count} unhealthy slices")

def split_data(data_dir):
    def move_patient_data(patient, image_dir, mask_dir):
        patient_image_dir = image_dir / patient.name
        patient_image_dir.mkdir(parents=True, exist_ok=True)

        if mask_dir:
            patient_mask_dir = mask_dir / patient.name
            patient_mask_dir.mkdir(parents=True, exist_ok=True)

        slice_files = [f for f in patient.iterdir() if f.is_file()]
        for slice_file in slice_files:
            target_image_path = patient_image_dir / slice_file.name
            slice_file.rename(target_image_path)

            if mask_dir:
                mask_file = Path(str(slice_file).replace("image", "mask"))
                target_mask_path = patient_mask_dir / mask_file.name
                mask_file.rename(target_mask_path)

        return len(slice_files)

    train_image_dir = Path(data_dir) / "train" / "image"
    val_image_dir = Path(data_dir) / "val" / "image"
    val_mask_dir = Path(data_dir) / "val" / "mask"
    test_image_dir = Path(data_dir) / "test" / "image"
    test_mask_dir = Path(data_dir) / "test" / "mask"

    train_image_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    val_mask_dir.mkdir(parents=True, exist_ok=True)
    test_image_dir.mkdir(parents=True, exist_ok=True)
    test_mask_dir.mkdir(parents=True, exist_ok=True)

    # healthy for train, val, test
    healthy_image_dir = Path(data_dir) / "healthy" / "image"
    healthy_patients = [patient for patient in healthy_image_dir.iterdir() if patient.is_dir()]
    random.shuffle(healthy_patients)

    val_slices = 0
    test_slices = 0
    for patient in healthy_patients:
        if val_slices < 20:
            moved_slices = move_patient_data(patient, val_image_dir, None)
            val_slices += moved_slices
        elif test_slices < 20:
            moved_slices = move_patient_data(patient, test_image_dir, None)
            test_slices += moved_slices
        else:
            moved_slices = move_patient_data(patient, train_image_dir, None)

    # unhealthy for val, test
    unhealthy_image_dir = Path(data_dir) / "unhealthy" / "image"
    unhealthy_patients = [patient for patient in unhealthy_image_dir.iterdir() if patient.is_dir()]
    random.shuffle(unhealthy_patients)

    val_slices = 0
    test_slices = 0
    for patient in unhealthy_patients:
        if val_slices < 15:
            moved_slices = move_patient_data(patient, val_image_dir, val_mask_dir)
            val_slices += moved_slices  
        elif test_slices < 15:
            moved_slices = move_patient_data(patient, test_image_dir, test_mask_dir)
            test_slices += moved_slices 

if __name__ == "__main__":
    dataset_url = {
        "2020": "https://www.med.upenn.edu/cbica/brats2020/data.html",
        "2021": "http://braintumorsegmentation.org/",
    }

    parser = argparse.ArgumentParser(description="Process data based on year")
    parser.add_argument("--year", "-y", default=2020, type=int, help="Specify the year")
    args = parser.parse_args()

    year = args.year
    print(f"BRATS-{year}")

    data_dir = f"brats-{year}/"

    if year == 2020:
        src_dir = os.path.join(data_dir, 'MICCAI_BraTS2020_TrainingData')
    elif year == 2021:
        src_dir = os.path.join(data_dir, 'BraTS2021_TrainingData')

    classify_healthy_data(data_dir, src_dir)
    split_data(data_dir)
