import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


import pyloudnorm as pyln

def _calculate_similarity(val1, val2, eps=1e-9):
    """
    Calculates a similarity score between two scalar values.
    The score ranges from 0 (very different) to 1 (identical).
    """
    abs_diff = abs(val1 - val2)
    # Use the mean of the absolute magnitude to normalize
    avg_magnitude = (abs(val1) + abs(val2)) / 2 + eps
    normalized_diff = abs_diff / avg_magnitude
    # np.exp(-x) maps a normalized difference [0, inf) to a similarity score (0, 1]
    return np.exp(-normalized_diff)


def get_hybrid_score(
    file1,
    file2,
    weights=None,
):
    if weights is None:
        weights = {
            "mel_spectrogram": 1,
            "lufs_envelope": 1,
            "mfcc": 1,
            "spectral_centroid": 1,
            "spectral_flatness": 1,
            "spectral_contrast": 1,
            "spectral_bandwidth": 1,
        }

    # --- 1. Chargement ---
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)


    # --- 2. Alignement sur l'attaque ---
    def align_on_onset(y, sr):
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        if len(onset_frames) > 0:
            onset_sample = librosa.frames_to_samples(onset_frames[0])
            return y[onset_sample:]
        return y

    # y1 = align_on_onset(y1, sr1)
    # y2 = align_on_onset(y2, sr2)

    # --- 3. Ajustement de longueur ---
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]

    # --- PARTIE A : SIMILARITÉ SPECTRALE ---
        # Remove the first 0.03 seconds to ignore the transient
    attack_seconds = 0.08
    attack_samples = int(attack_seconds * sr1)
    y1_mel = y1[attack_samples:]
    y2_mel = y2[attack_samples:]

    S1 = librosa.feature.melspectrogram(y=y1_mel, sr=sr1, n_mels=128, hop_length=50)
    S2 = librosa.feature.melspectrogram(y=y2_mel, sr=sr2, n_mels=128, hop_length=50)

    S1_db = librosa.power_to_db(S1, ref=np.max)
    S2_db = librosa.power_to_db(S2, ref=np.max)

    # MOYENNE TEMPORELLE pour plus de robustesse
    S1_vec = np.mean(S1_db, axis=1).reshape(1, -1)
    S2_vec = np.mean(S2_db, axis=1).reshape(1, -1)

    score_spec = cosine_similarity(S1_vec, S2_vec)[0][0]

    # Sécurisation des valeurs
    score_spec = float(np.clip(score_spec, -1, 1))
    if np.isnan(score_spec):
        score_spec = 0.0

    # --- PARTIE B : SIMILARITÉ D’ENVELOPPE LUFS ---

    def get_lufs_envelope(y, sr):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)       
        return loudness

    env1 = get_lufs_envelope(y1, sr1)
    env2 = get_lufs_envelope(y2, sr2)
    score_env = _calculate_similarity(env1, env2)
    if np.isnan(score_env):
        score_env = 0.0

    # --- PARTIE C : AUTRES MÉTRIQUES SPECTRALES ---

    # MFCC
    mfcc1 = np.mean(librosa.feature.mfcc(y=y1, sr=sr1), axis=1).reshape(1, -1)
    mfcc2 = np.mean(librosa.feature.mfcc(y=y2, sr=sr2), axis=1).reshape(1, -1)
    score_mfcc = float(np.clip(cosine_similarity(mfcc1, mfcc2)[0][0], -1, 1))

    # Spectral Centroid
    centroid1 = np.mean(librosa.feature.spectral_centroid(y=y1, sr=sr1))
    centroid2 = np.mean(librosa.feature.spectral_centroid(y=y2, sr=sr2))
    score_centroid = _calculate_similarity(centroid1, centroid2)

    # Spectral Flatness
    flatness1 = np.mean(librosa.feature.spectral_flatness(y=y1))
    flatness2 = np.mean(librosa.feature.spectral_flatness(y=y2))
    score_flatness = _calculate_similarity(flatness1, flatness2)

    # Spectral Contrast
    contrast1 = np.mean(librosa.feature.spectral_contrast(y=y1, sr=sr1), axis=1).reshape(1, -1)
    contrast2 = np.mean(librosa.feature.spectral_contrast(y=y2, sr=sr2), axis=1).reshape(1, -1)
    score_contrast = float(np.clip(cosine_similarity(contrast1, contrast2)[0][0], -1, 1))

    # Spectral Bandwidth
    bandwidth1 = np.mean(librosa.feature.spectral_bandwidth(y=y1, sr=sr1))
    bandwidth2 = np.mean(librosa.feature.spectral_bandwidth(y=y2, sr=sr2))
    score_bandwidth = _calculate_similarity(bandwidth1, bandwidth2)

    # --- Rassemblement des scores ---
    scores = {
        "mel_spectrogram": score_spec,
        "lufs_envelope": score_env,
        "mfcc": score_mfcc,
        "spectral_centroid": score_centroid,
        "spectral_flatness": score_flatness,
        "spectral_contrast": score_contrast,
        "spectral_bandwidth": score_bandwidth,
    }

    # --- SCORE FINAL ---
    weighted_score_sum = sum(scores[k] * v for k, v in weights.items())
    weight_sum = sum(weights.values())

    final_score = weighted_score_sum / weight_sum if weight_sum > 0 else 0.0

    # Passage en pourcentage
    result = {
        "total": round(float(final_score * 100), 2),
        "details": {
            key: round(float(value * 100), 2) for key, value in scores.items()
        }
    }

    details_str = ", ".join(f"{k}: {v:.2f}%" for k, v in result["details"].items())
    print(
        f"|{os.path.basename(file1)}| VS |{os.path.basename(file2)}| → "
        f"Total: {result['total']:.2f}% ({details_str})"
    )

    return result
