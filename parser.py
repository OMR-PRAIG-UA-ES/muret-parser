import os
import json
import argparse
import requests
from io import BytesIO

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold, train_test_split


def parser_args():
    parser = argparse.ArgumentParser(
        description="MuRET dataset parser. Converts a full page sample (multiple staves) into multiple individual samples, one for each staff.\
            Creates a new folder (output_folder_path) with the new image samples (output_folder_path/Images) and their corresponding sequences (output_folder_path/GT).\
                Creates k-folds (output_folder_path/Folds) for the dataset (Train: 60%, Validation: 20%, Test: 20%)."
    )
    parser.add_argument(
        "--muret_json_folder_path",
        type=str,
        required=True,
        help="Path to the folder containing the JSON files downloaded from MuRET.",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        required=True,
        help="Path to the folder where the new samples and their corresponding sequences will be saved.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds to create for the dataset.",
    )
    return parser.parse_args()


def parsing_muret_json(muret_json_folder_path: str, output_folder_path: str):
    """
    Converts a full page sample (multiple staves) into multiple individual samples, one for each staff.
    Creates a new folder with the new samples and their corresponding sequences.

    Args:
        - muret_json_folder_path (str): path to the folder containing the JSON files downloaded from MuRET.
        - output_folder_path (str): path to the folder where the new samples and their corresponding sequences will be saved.

    The output folder will have the following structure:
        Output_folder_path
        ├── Images
        │   ├── sample1.jpg
        │   ├── sample2.jpg
        │   └── ...
        └── GT
            ├── sample1.jpg.txt
            ├── sample2.jpg.txt
            └── ...
    """

    def download_image(image_url: str) -> np.ndarray | None:
        """
        Download an image from a URL and return it as a numpy array.
        Args:
            image_url (str): URL of the image to download.
        Returns:
            np.ndarray | None: image as a numpy array or None if the download failed.
        """
        response = requests.get(image_url)
        if response.status_code == 200:
            image_bytes = BytesIO(response.content)
            img = Image.open(image_bytes)
            return np.array(img)
        return None

    # Create output folders
    img_folder_path = os.path.join(output_folder_path, "Images")
    gt_folder_path = os.path.join(output_folder_path, "GT")
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(img_folder_path, exist_ok=True)
    os.makedirs(gt_folder_path, exist_ok=True)

    # Iterate over JSON files
    for root, dirs, files in os.walk(muret_json_folder_path):
        for file in files:
            if (
                file.endswith(".json")
                and not file.startswith(".")
                and not file.startswith("dictionary")
            ):
                # Load JSON file
                file = os.path.join(root, file)
                with open(file) as json_file:
                    data = json.load(json_file)

                # Get page name
                img_name = data["filename"].replace(".JPG", ".jpg")

                # Download full page image
                img_path = data.get("original", None)
                if img_path is None:
                    img_path = data.get("url", None)
                assert img_path is not None, f"No image path found for JSON file {file}"
                img = download_image(img_path)

                # Iterate over staves
                for page in data["pages"]:
                    if "regions" in page:
                        for region in page["regions"]:
                            if region["type"] == "staff" and "symbols" in region:
                                symbols = region["symbols"]
                                if len(symbols) > 0:
                                    # Image:
                                    # 1. Get staff bounding box
                                    left, right, top, bottom = region[
                                        "bounding_box"
                                    ].values()
                                    # 2. There are some negative values: put them to 0
                                    left = max(0, left)
                                    top = max(0, top)
                                    # 3. Crop image
                                    staff_img = img[top:bottom, left:right]
                                    # 4. Save image
                                    staff_name = os.path.join(
                                        img_folder_path,
                                        img_name.replace(
                                            ".jpg", f"_{region['id']}.jpg"
                                        ),
                                    )
                                    staff_img = Image.fromarray(staff_img)
                                    staff_img.save(staff_name)

                                    # Ground truth:
                                    gt = [
                                        f"{s['agnostic_symbol_type']}:{s['position_in_staff']}"
                                        for s in symbols
                                    ]
                                    gt = " ".join(gt).strip()
                                    # 1. Save ground truth
                                    gt_name = os.path.join(
                                        gt_folder_path,
                                        img_name.replace(
                                            ".jpg", f"_{region['id']}.jpg.txt"
                                        ),
                                    )
                                    with open(gt_name, "w") as f:
                                        f.write(gt)


def create_kfolds(output_folder_path: str, k: int = 5):
    """
    Create k-folds for the dataset.
    Train: 60%, Validation: 20%, Test: 20%

    Args:
        - output_folder_path (str): path to the folder containing the samples and their corresponding sequences.
        - k (int): number of folds to create.

    The output folder will have the following structure:
        Output_folder_path
        ├── Folds
        │   ├── train_gt_fold0.dat
        │   ├── val_gt_fold0.dat
        │   ├── test_gt_fold0.dat
        │   ├── ...
        └── ...
    """

    # Get all samples
    img_folder_path = os.path.join(output_folder_path, "Images")
    samples = [
        f
        for f in os.listdir(img_folder_path)
        if f.endswith(".jpg") and not f.startswith(".")
    ]
    samples = np.array(samples)

    # Create output folder
    output_dir = os.path.join(output_folder_path, "Folds")
    os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=k, random_state=42, shuffle=True)
    for id, (train_index, test_index) in enumerate(kf.split(samples)):
        train_fold = os.path.join(output_dir, f"train_gt_fold{id}.dat")
        val_fold = os.path.join(output_dir, f"val_gt_fold{id}.dat")
        test_fold = os.path.join(output_dir, f"test_gt_fold{id}.dat")

        X_train, X_test = samples[train_index], samples[test_index]
        X_train, X_val = train_test_split(
            X_train, test_size=0.25, random_state=42
        )  # 0.25 x 0.8 = 0.2

        with open(train_fold, "w") as txt:
            for img in X_train:
                txt.write(img + "\n")
        with open(val_fold, "w") as txt:
            for img in X_val:
                txt.write(img + "\n")
        with open(test_fold, "w") as txt:
            for img in X_test:
                txt.write(img + "\n")


if __name__ == "__main__":
    args = parser_args()
    parsing_muret_json(
        muret_json_folder_path=args.muret_json_folder_path,
        output_folder_path=args.output_folder_path,
    )
    create_kfolds(output_folder_path=args.output_folder_path, k=args.k)
