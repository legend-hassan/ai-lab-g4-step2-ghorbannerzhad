"""Audio loading and compact, fixed-length feature extraction."""
from pathlib import Path
import numpy as np
import librosa


def extract_features(path: str | Path, sr: int = 22050, duration: float = 30.0,
                     n_mfcc: int = 20) -> np.ndarray:
    """Extract aggregated MFCC and spectral features from one audio file."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    if y.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {path}")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = [mfcc.mean(axis=1), mfcc.std(axis=1),
                np.array([librosa.feature.rms(y=y).mean(),
                          librosa.feature.zero_crossing_rate(y).mean(),
                          librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
                          librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()])]
    vector = np.concatenate(features).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite features extracted from: {path}")
    return vector


def load_folder_dataset(root: str | Path):
    """Load class-folder data: root/genre/audio files. Returns X, y, paths."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    genres = sorted(p for p in root.iterdir() if p.is_dir())
    if len(genres) < 2:
        raise ValueError("Expected at least two genre subdirectories")
    X, y, paths = [], [], []
    for genre_dir in genres:
        for path in sorted(genre_dir.glob("*")):
            if path.suffix.lower() not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                continue
            X.append(extract_features(path))
            y.append(genre_dir.name)
            paths.append(path)
    if not X:
        raise ValueError("No supported audio files found in genre folders")
    return np.vstack(X), np.asarray(y), paths
