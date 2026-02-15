"""Evaluate generated images using the Kvasir classifier."""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from medsteer.classifier import load_classifier, KVASIR_LABELS


def evaluate_generated_images(
    images_dir: str,
    raw_csv_path: str,
    checkpoint_path: str,
    model_name: str = "convnext_large",
):
    """
    Evaluate generated images against ground truth labels using the classifier.

    Args:
        images_dir: Directory containing generated images.
        raw_csv_path: Path to raw.csv with (file_name, text) columns.
        checkpoint_path: Path to classifier checkpoint.
        model_name: Classifier model architecture name.

    Returns:
        dict with accuracy, accuracy_threshold, f1, and auc metrics.
    """
    label_to_idx = {label: i for i, label in enumerate(KVASIR_LABELS)}
    prefix = "An endoscopic image of "

    raw_df = pd.read_csv(raw_csv_path)
    file_to_label = {}
    for _, row in raw_df.iterrows():
        label_str = row["text"].replace(prefix, "").strip()
        if label_str in label_to_idx:
            base_name = os.path.splitext(row["file_name"])[0]
            file_to_label[base_name] = label_to_idx[label_str]

    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    data = []
    for img_file in image_files:
        matched_label = None
        for base_name, label in file_to_label.items():
            if img_file.startswith(base_name):
                matched_label = label
                break
        if matched_label is not None:
            data.append(
                {
                    "path": os.path.join(images_dir, img_file),
                    "label": matched_label,
                }
            )

    print(f"Found {len(data)} matched images out of {len(image_files)} total images.")

    if not data:
        print("No matches found. Check image naming convention.")
        return None

    classifier = load_classifier(checkpoint_path, model_name=model_name)
    classifier.model.eval()
    device = classifier.device

    all_preds = []
    all_probs = []
    all_labels = []

    for item in tqdm(data, desc="Running inference"):
        img = Image.open(item["path"])
        label = item["label"]

        tensor = classifier.preprocess(img).to(device)
        with torch.no_grad():
            logits = classifier.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

        all_preds.append(pred.item())
        all_probs.append(probs[0].cpu().numpy())
        all_labels.append(label)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)

    correct_class_probs = all_probs[np.arange(len(all_labels)), all_labels]
    acc_threshold = (correct_class_probs > 0.5).mean()

    f1 = f1_score(all_labels, all_preds, average="weighted")

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        auc = 0.0

    results = {
        "accuracy": acc,
        "accuracy_threshold": acc_threshold,
        "f1": f1,
        "auc": auc,
    }

    print("\n" + "=" * 30)
    print("Metrics for generated images:")
    print(f"Accuracy (Argmax): {acc:.4f}")
    print(f"Accuracy (Threshold 0.5): {acc_threshold:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"AUC (macro): {auc:.4f}")
    print("=" * 30)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images with Kvasir classifier.")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of generated images.")
    parser.add_argument("--raw_csv_path", type=str, required=True, help="Path to raw.csv.")
    parser.add_argument("--classifier_ckpt", type=str, required=True, help="Classifier checkpoint path.")
    parser.add_argument("--model_name", type=str, default="convnext_large", help="Classifier model name.")
    args = parser.parse_args()

    evaluate_generated_images(
        images_dir=args.images_dir,
        raw_csv_path=args.raw_csv_path,
        checkpoint_path=args.classifier_ckpt,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
