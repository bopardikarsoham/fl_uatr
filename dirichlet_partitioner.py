"""
ShipsEar 5-Class Dataset Preparation for Federated Learning (PFLlib + ResNet-18)
=================================================================================

Expected dataset structure (processed 5-class version):
  shipsear_5s_16k/
    0/
      *.wav
    1/
      *.wav
    2/
      *.wav
    3/
      *.wav
    4/
      *.wav

Class meaning (from explore_shipsear.py):
  0 -> A : Small Working
  1 -> B : Small Rec/Utility
  2 -> C : Passenger Ferries
  3 -> D : Large Commercial
  4 -> E : Background Noise

Output structure:
  output_root/
    niid_alpha{alpha}_c{n_clients}/
      train/
        0.npz
        1.npz
        ...
      test/
        0.npz
        1.npz
        ...
      partition_stats.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------------------------------
# 5-class ShipsEar setup
# -------------------------------------------------
SHIPSEAR_CLASSES = ["0", "1", "2", "3", "4"]

CLASS_DESCRIPTIONS = {
    "0": "A - Small Working (Fishing, Trawler, Mussel, Tug, Dredger)",
    "1": "B - Small Rec/Utility (Motorboat, Pilot, Sailboat)",
    "2": "C - Passenger Ferries",
    "3": "D - Large Commercial (Ocean Liner, Ro-Ro)",
    "4": "E - Background Noise",
}

# -------------------------------------------------
# Audio -> mel spectrogram parameters
# -------------------------------------------------
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
IMG_SIZE = 224
SEGMENT_SECS = 5.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_segment(
    wav_path: str,
    target_sr: int = SAMPLE_RATE,
    segment_secs: float = SEGMENT_SECS,
):
    """
    Load a .wav file, resample if needed, mix to mono,
    and split into fixed-length segments.

    Returns:
        list[torch.Tensor]: each tensor shape = (1, segment_len)
    """
    waveform, sr = torchaudio.load(wav_path)

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    segment_len = int(target_sr * segment_secs)
    total_len = waveform.shape[-1]
    segments = []

    if total_len <= segment_len:
        # Pad short clip
        padded = torch.zeros((1, segment_len), dtype=waveform.dtype)
        padded[:, :total_len] = waveform[:, :total_len]
        segments.append(padded)
    else:
        # Non-overlapping fixed-length segments
        n_full = total_len // segment_len
        for i in range(n_full):
            start = i * segment_len
            end = start + segment_len
            segments.append(waveform[:, start:end])

    return segments


def waveform_to_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    img_size: int = IMG_SIZE,
):
    """
    Convert waveform to a 3-channel mel spectrogram image for ResNet.
    Output shape: (3, img_size, img_size), float32
    """
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )(waveform)  # (1, n_mels, time)

    mel_db = T.AmplitudeToDB(stype="power")(mel)

    # Normalize per sample to [0, 1]
    mel_min = mel_db.amin(dim=(-2, -1), keepdim=True)
    mel_max = mel_db.amax(dim=(-2, -1), keepdim=True)
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

    # Resize to square image
    mel_resized = torch.nn.functional.interpolate(
        mel_norm.unsqueeze(0),  # (1, 1, n_mels, time)
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # (1, img_size, img_size)

    # Repeat channel to 3x for ResNet
    img = mel_resized.repeat(3, 1, 1)  # (3, img_size, img_size)
    return img.cpu().numpy().astype(np.float32)


def build_dataset(
    data_root: str,
    class_names: list[str],
    sample_rate: int = SAMPLE_RATE,
    segment_secs: float = SEGMENT_SECS,
    img_size: int = IMG_SIZE,
):
    """
    Walk through class folders and build full dataset arrays.

    Returns:
        X: np.ndarray of shape (N, 3, img_size, img_size)
        y: np.ndarray of shape (N,)
    """
    data_root = Path(data_root)
    X, y = [], []

    print("Scanning class folders:")
    for label, cls_name in enumerate(class_names):
        cls_dir = data_root / cls_name
        if not cls_dir.exists():
            print(f"  [WARNING] Missing class folder: {cls_dir}")
            continue

        # recursive scan
        wav_files = sorted(cls_dir.rglob("*.wav"))
        print(f"  Class {cls_name}: {len(wav_files)} wav files")

        for wav_path in tqdm(wav_files, desc=f"Processing class {cls_name}", leave=False):
            try:
                segments = load_and_segment(
                    str(wav_path),
                    target_sr=sample_rate,
                    segment_secs=segment_secs,
                )
                for seg in segments:
                    img = waveform_to_spectrogram(
                        seg,
                        sample_rate=sample_rate,
                        img_size=img_size,
                    )
                    X.append(img)
                    y.append(label)
            except Exception as e:
                print(f"[WARNING] Failed on {wav_path}: {e}")

    if len(X) == 0:
        raise RuntimeError(
            "No samples were created. Check --data_root and make sure it contains "
            "class folders 0,1,2,3,4 with .wav files."
        )

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def dirichlet_partition(
    y: np.ndarray,
    n_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_samples: int = 1,
):
    """
    Partition class indices across clients using a Dirichlet distribution.
    Returns:
        list[list[int]]: sample indices for each client
    """
    n_classes = len(np.unique(y))
    client_idxs = [[] for _ in range(n_clients)]

    for cls in range(n_classes):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        proportions = rng.dirichlet([alpha] * n_clients)
        splits = (proportions * len(cls_indices)).astype(int)

        # fix rounding
        splits[-1] = len(cls_indices) - splits[:-1].sum()

        start = 0
        for client_id, count in enumerate(splits):
            client_idxs[client_id].extend(cls_indices[start:start + count].tolist())
            start += count

    for i, idxs in enumerate(client_idxs):
        if len(idxs) < min_samples:
            print(
                f"[WARNING] Client {i} has only {len(idxs)} samples "
                f"(min={min_samples}). Consider fewer clients or larger alpha."
            )

    return client_idxs


def save_pfllib_splits(
    X: np.ndarray,
    y: np.ndarray,
    client_idxs: list[list[int]],
    output_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Save per-client train/test splits in PFLlib-compatible format:
      output_dir/train/{client_id}.npz
      output_dir/test/{client_id}.npz
    """
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for client_id, idxs in enumerate(client_idxs):
        idxs = np.array(idxs, dtype=np.int64)

        if len(idxs) == 0:
            print(f"[WARNING] Client {client_id} has 0 samples, skipping.")
            continue

        X_client = X[idxs]
        y_client = y[idxs]

        # Stratified split where possible
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_client,
                y_client,
                test_size=test_ratio,
                stratify=y_client,
                random_state=seed,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_client,
                y_client,
                test_size=test_ratio,
                random_state=seed,
            )

        np.savez_compressed(train_dir / f"{client_id}.npz", data=X_tr, targets=y_tr)
        np.savez_compressed(test_dir / f"{client_id}.npz", data=X_te, targets=y_te)

        class_dist = {int(c): int((y_client == c).sum()) for c in np.unique(y_client)}
        stats[f"client_{client_id}"] = {
            "total_samples": int(len(idxs)),
            "train_samples": int(len(y_tr)),
            "test_samples": int(len(y_te)),
            "class_distribution": class_dist,
        }

        print(
            f"  Client {client_id:2d}: "
            f"{len(y_tr):4d} train  {len(y_te):3d} test  "
            f"classes={sorted(class_dist.keys())}"
        )

    stats_path = Path(output_dir) / "partition_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nPartition stats saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 5-class ShipsEar dataset for FL experiments with PFLlib + ResNet-18"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to ShipsEar processed root directory containing folders 0,1,2,3,4",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/shipsear_fl",
        help="Root directory for output partitions",
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Number of FL clients to generate (e.g. 5 10 20)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="Dirichlet alpha values. Lower = more heterogeneous.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of each client's data used for local test split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Target sample rate (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--segment_secs",
        type=float,
        default=SEGMENT_SECS,
        help=f"Segment length in seconds (default: {SEGMENT_SECS})",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=IMG_SIZE,
        help=f"Image size for ResNet input (default: {IMG_SIZE})",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    print("\n[1/3] Building full dataset (audio -> mel spectrogram)...")
    print(f"      Data root : {args.data_root}")
    print(f"      Classes   : {SHIPSEAR_CLASSES}")
    print(f"      SR={args.sample_rate}Hz  segment={args.segment_secs}s  img={args.img_size}x{args.img_size}")
    print("      Class map :")
    for c in SHIPSEAR_CLASSES:
        print(f"        {c}: {CLASS_DESCRIPTIONS[c]}")
    print()

    X, y = build_dataset(
        args.data_root,
        SHIPSEAR_CLASSES,
        sample_rate=args.sample_rate,
        segment_secs=args.segment_secs,
        img_size=args.img_size,
    )

    print(f"\nDataset built: {X.shape[0]} samples, shape={X.shape}")
    class_counts = {c: int((y == i).sum()) for i, c in enumerate(SHIPSEAR_CLASSES)}
    print("Class distribution:", class_counts)

    total_configs = len(args.n_clients) * len(args.alphas)
    print(f"\n[2/3] Generating {total_configs} partition configurations...\n")

    for n_clients in args.n_clients:
        for alpha in args.alphas:
            tag = f"niid_alpha{alpha}_c{n_clients}"
            output_dir = Path(args.output_root) / tag

            print(f"--- Config: {n_clients} clients, alpha={alpha} -> {output_dir}")
            rng = np.random.default_rng(args.seed)
            client_idxs = dirichlet_partition(y, n_clients, alpha, rng=rng)

            save_pfllib_splits(
                X,
                y,
                client_idxs,
                str(output_dir),
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
            print()

    print("[3/3] Done. All partitions saved under:", args.output_root)
    print("\nPFLlib usage:")
    print("  Set dataset path to one config folder, for example:")
    print(f"    {args.output_root}/niid_alpha0.1_c10/")
    print("  Each client file contains:")
    print("    data    -> float32, shape (N, 3, img_size, img_size)")
    print("    targets -> int64,   shape (N,)")

if __name__ == "__main__":
    Main()