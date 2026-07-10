"""
King Aim — GRU training data preparation.

Collects all gru_sequences.json files from King Aim's session log folders
(runs/logs/session_*/gru_sequences.json), filters short/low-quality sequences,
and splits them into train/ and val/ subdirectories ready for train_gru.py.

Usage:
    python prepare_gru_data.py
    python prepare_gru_data.py --logs-dir runs/logs --out-dir data/tracks --val-fraction 0.15 --min-frames 12
"""

import json, os, random, math, shutil, argparse, glob
from pathlib import Path


def load_all_sequences(logs_dir, min_frames):
    sequences = []
    pattern = str(Path(logs_dir) / "session_*" / "gru_sequences.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No gru_sequences.json found under {logs_dir}")
        return sequences
    for path in files:
        with open(path) as f:
            data = json.load(f)
        kept = [s for s in data if len(s.get("frames", [])) >= min_frames]
        print(f"  {Path(path).parent.name}: {len(data)} tracks  →  {len(kept)} kept (≥{min_frames} frames)")
        sequences.extend(kept)
    return sequences


def split(sequences, val_fraction, seed=42):
    rng = random.Random(seed)
    rng.shuffle(sequences)
    n_val = max(1, math.ceil(len(sequences) * val_fraction))
    return sequences[n_val:], sequences[:n_val]


def write_split(sequences, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / "gru_sequences.json"
    with open(out_path, "w") as f:
        json.dump(sequences, f, separators=(",", ":"))
    print(f"  {name}: {len(sequences)} sequences  →  {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare GRU training data from session logs")
    parser.add_argument("--logs-dir",     default="runs/logs",
                        help="Root directory containing session_* folders (default: runs/logs)")
    parser.add_argument("--out-dir",      default="data/tracks",
                        help="Output root; train/ and val/ subdirs are created here (default: data/tracks)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Fraction of sequences to hold out for validation (default: 0.15)")
    parser.add_argument("--min-frames",   type=int,   default=12,
                        help="Minimum frames per track; shorter sequences are dropped (default: 12, need ≥9+1)")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    print(f"Loading sequences from {args.logs_dir}  (min_frames={args.min_frames})")
    sequences = load_all_sequences(args.logs_dir, args.min_frames)
    if not sequences:
        print("No sequences loaded — check --logs-dir or enable Detection Logging in King Aim.")
        return

    print(f"\nTotal: {len(sequences)} sequences")
    train_seqs, val_seqs = split(sequences, args.val_fraction, args.seed)

    print(f"\nWriting split:")
    write_split(train_seqs, os.path.join(args.out_dir, "train"), "train")
    write_split(val_seqs,   os.path.join(args.out_dir, "val"),   "val")

    print(f"\nDone.  Next step:")
    print(f"  python train_gru.py")


if __name__ == "__main__":
    main()
