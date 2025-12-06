import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


import pyloudnorm as pyln


def get_hybrid_score(
        file1, file2,
        weights=(0.6, 0.4),
        verbose=False
    ):

    w_spec, w_env = weights

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

    S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128)
    S2 = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128)

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

    def loudness_similarity(loudness1, loudness2, eps=1e-9):
        """
        Calcule un score de similarité entre deux valeurs de loudness (LUFS).
        Le score va de 0 (très différent) à 1 (identique).
        Les valeurs LUFS sont négatives, mais la logique reste la même.
        """
        abs_diff = abs(loudness1 - loudness2)
        # Utilise la moyenne de la magnitude absolue pour normaliser
        avg_magnitude = (abs(loudness1) + abs(loudness2)) / 2 + eps
        normalized_diff = abs_diff / avg_magnitude
        # np.exp(-x) mappe une différence normalisée [0, inf) vers un score de similarité (0, 1]
        return np.exp(-normalized_diff)

    def get_lufs_envelope(y, sr):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)       
        return loudness

    env1 = get_lufs_envelope(y1, sr1)
    env2 = get_lufs_envelope(y2, sr2)
    score_env = loudness_similarity(env1, env2)
    if np.isnan(score_env):
        score_env = 0.0

    # --- SCORE FINAL ---
    final_score = w_spec * score_spec + w_env * score_env

    # Passage en pourcentage
    result = {
        "total": float(final_score * 100),
        "detail_spectre": float(score_spec * 100),
        "detail_enveloppe": float(score_env * 100),
    }

    if verbose:
        print(f"{os.path.basename(file1)} vs {os.path.basename(file2)} → {result}")

    return result
