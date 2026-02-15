"""Compute FID between a generated image set and a reference set."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm


class FeatureStats:
    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_outer = np.zeros((dim, dim), dtype=np.float64)

    def update(self, features):
        features = features.astype(np.float64)
        self.count += features.shape[0]
        self.sum += features.sum(axis=0)
        self.sum_outer += features.T @ features

    def finalize(self):
        if self.count < 2:
            raise ValueError("Need at least 2 samples to compute covariance.")
        mean = self.sum / self.count
        cov = (self.sum_outer - self.count * np.outer(mean, mean)) / (self.count - 1)
        return mean, cov


class InceptionFeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        try:
            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
        except Exception:
            model = inception_v3(pretrained=True, aux_logits=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)

    def preprocess(self, image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    def extract(self, images):
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
        if hasattr(feats, "logits"):
            feats = feats.logits
        elif isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.cpu().numpy()


def _list_images(root, exts, recursive=False):
    root = Path(root)
    if recursive:
        paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    else:
        paths = [p for p in root.glob("*") if p.suffix.lower() in exts]
    return sorted(paths)


def _filter_paths(paths, pattern):
    if not pattern:
        return paths
    regex = re.compile(pattern)
    return [p for p in paths if regex.search(p.name)]


def _compute_stats(paths, extractor, batch_size):
    stats = None
    batch = []

    iterator = tqdm(paths, desc="Computing stats") if len(paths) > 100 else paths

    for path in iterator:
        try:
            image = Image.open(path)
            try:
                batch.append(extractor.preprocess(image))
            finally:
                image.close()
        except Exception as e:
            print(f"Warning: Could not open {path}: {e}")
            continue

        if len(batch) >= batch_size:
            feats = extractor.extract(batch)
            if stats is None:
                stats = FeatureStats(feats.shape[1])
            stats.update(feats)
            batch = []
    if batch:
        feats = extractor.extract(batch)
        if stats is None:
            stats = FeatureStats(feats.shape[1])
        stats.update(feats)

    if stats is None:
        raise ValueError("No valid images processed.")

    return stats.finalize()


def _sqrtm(matrix):
    try:
        from scipy import linalg
        return linalg.sqrtm(matrix)
    except Exception:
        eigvals, eigvecs = np.linalg.eig(matrix)
        eigvals = np.where(eigvals < 0, 0, eigvals)
        sqrt_eigvals = np.sqrt(eigvals)
        return eigvecs @ np.diag(sqrt_eigvals) @ np.linalg.inv(eigvecs)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Frechet distance (FID) between two Gaussians."""
    diff = mu1 - mu2
    covmean = _sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def load_stats(path):
    """Load precomputed mean/cov stats from a .npz file."""
    data = np.load(path)
    return data["mu"], data["sigma"]


def save_stats(path, mu, sigma):
    """Save mean/cov stats to a .npz file."""
    np.savez(path, mu=mu, sigma=sigma)


def compute_fid(
    gen_dir=None,
    ref_dir=None,
    gen_paths=None,
    ref_paths=None,
    batch_size=32,
    device="auto",
    recursive=False,
    gen_pattern=None,
    ref_pattern=None,
    max_items=None,
):
    """
    Compute FID between generated and reference image sets.

    Args:
        gen_dir: Directory of generated images.
        ref_dir: Directory of reference images.
        gen_paths: Explicit list of generated image paths (overrides gen_dir).
        ref_paths: Explicit list of reference image paths (overrides ref_dir).
        batch_size: Batch size for feature extraction.
        device: Device ("auto", "cuda", "cpu").
        recursive: Search subdirectories.
        gen_pattern: Regex to filter generated images.
        ref_pattern: Regex to filter reference images.
        max_items: Limit number of images.

    Returns:
        float: FID score.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = InceptionFeatureExtractor(device)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    if ref_paths is None:
        ref_paths = _list_images(ref_dir, exts, recursive=recursive)
        ref_paths = _filter_paths(ref_paths, ref_pattern)
    if max_items:
        ref_paths = ref_paths[:max_items]
    if not ref_paths:
        raise ValueError("No reference images found.")

    if gen_paths is None:
        gen_paths = _list_images(gen_dir, exts, recursive=recursive)
        gen_paths = _filter_paths(gen_paths, gen_pattern)
    if max_items:
        gen_paths = gen_paths[:max_items]
    if not gen_paths:
        raise ValueError("No generated images found.")

    print(f"Computing stats for {len(ref_paths)} reference images...")
    mu_ref, sigma_ref = _compute_stats(ref_paths, extractor, batch_size)

    print(f"Computing stats for {len(gen_paths)} generated images...")
    mu_gen, sigma_gen = _compute_stats(gen_paths, extractor, batch_size)

    fid = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
    print(f"FID: {fid:.4f}")
    return fid


def main():
    """CLI entry point for FID computation."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute FID between generated and reference images.")
    parser.add_argument("--csv_dir", type=str, help="Directory containing CSV files with image paths.")
    parser.add_argument("--gen_dir", type=str, help="Generated images directory.")
    parser.add_argument("--ref_dir", type=str, default=None, help="Reference images directory.")
    parser.add_argument("--gen_pattern", type=str, default=None, help="Regex to filter generated images.")
    parser.add_argument("--ref_pattern", type=str, default=None, help="Regex to filter reference images.")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--ref_stats", type=str, default=None, help="Load reference stats from npz.")
    parser.add_argument("--gen_stats", type=str, default=None, help="Load generated stats from npz.")
    parser.add_argument("--save_ref_stats", type=str, default=None, help="Save reference stats to npz.")
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()

    import torchvision
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    if not args.csv_dir and not args.gen_dir and not args.gen_stats and not args.save_ref_stats:
        raise SystemExit("Must provide either --csv_dir or --gen_dir (or --gen_stats).")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    extractor = None
    def get_extractor():
        nonlocal extractor
        if extractor is None:
            extractor = InceptionFeatureExtractor(device)
        return extractor

    # Reference stats
    if args.ref_stats:
        mu_ref, sigma_ref = load_stats(args.ref_stats)
        print(f"Loaded reference stats from {args.ref_stats}")
    else:
        if not args.ref_dir:
            raise SystemExit("--ref_dir or --ref_stats is required for FID.")
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        ref_paths = _list_images(args.ref_dir, exts, recursive=args.recursive)
        ref_paths = _filter_paths(ref_paths, args.ref_pattern)
        if args.max_items:
            ref_paths = ref_paths[:args.max_items]
        if not ref_paths:
            raise SystemExit("No reference images found.")
        print(f"Computing stats for {len(ref_paths)} reference images...")
        mu_ref, sigma_ref = _compute_stats(ref_paths, get_extractor(), args.batch_size)
        if args.save_ref_stats:
            save_stats(args.save_ref_stats, mu_ref, sigma_ref)
            print(f"Saved reference stats to {args.save_ref_stats}")

    results = []

    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        csv_files = sorted(list(csv_dir.glob("*.csv")))
        if not csv_files:
            print(f"No CSV files found in {csv_dir}")
            return
        print(f"Found {len(csv_files)} CSV files in {csv_dir}")
        for csv_file in csv_files:
            print(f"\nProcessing {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
            if 'path' not in df.columns:
                print(f"Skipping {csv_file.name}: 'path' column not found.")
                continue
            gen_paths = df['path'].tolist()
            if args.max_items:
                gen_paths = gen_paths[:args.max_items]
            if not gen_paths:
                print(f"No paths in {csv_file.name}")
                continue
            valid_paths = [p for p in gen_paths if os.path.exists(p)]
            if len(valid_paths) < len(gen_paths):
                print(f"Warning: {len(gen_paths) - len(valid_paths)} images not found. Using {len(valid_paths)} valid images.")
            if len(valid_paths) < 2:
                print(f"Skipping {csv_file.name}: Need at least 2 images for FID.")
                continue
            try:
                mu_gen, sigma_gen = _compute_stats(valid_paths, get_extractor(), args.batch_size)
                fid_value = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
                print(f"FID for {csv_file.name}: {fid_value:.4f}")
                results.append({"file": csv_file.name, "fid": fid_value, "n_images": len(valid_paths)})
            except Exception as e:
                print(f"Error computing FID for {csv_file.name}: {e}")

    elif args.gen_dir or args.gen_stats:
        if args.gen_stats:
            mu_gen, sigma_gen = load_stats(args.gen_stats)
            print(f"Loaded generated stats from {args.gen_stats}")
            fid_value = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
            print(f"FID: {fid_value:.4f}")
        else:
            exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            gen_paths = _list_images(args.gen_dir, exts, recursive=args.recursive)
            gen_paths = _filter_paths(gen_paths, args.gen_pattern)
            if args.max_items:
                gen_paths = gen_paths[:args.max_items]
            if not gen_paths:
                print("No generated images found.")
            else:
                print(f"Computing stats for {len(gen_paths)} generated images...")
                mu_gen, sigma_gen = _compute_stats(gen_paths, get_extractor(), args.batch_size)
                fid_value = frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
                print(f"FID: {fid_value:.4f}")

    if results:
        print("\n" + "=" * 50)
        print("Summary of FID Scores")
        print("=" * 50)
        print(f"{'File':<40} {'FID':>10} {'N_imgs':>8}")
        print("-" * 50)
        for res in results:
            print(f"{res['file']:<40} {res['fid']:>10.4f} {res['n_images']:>8}")


if __name__ == "__main__":
    main()
