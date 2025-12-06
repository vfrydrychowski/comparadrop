import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


def get_hybrid_score(
        file1, file2,
        weights=(0.6, 0.4),
        normalize=False,
        verbose=False
    ):

    w_spec, w_env = weights

    # --- 1. Chargement ---
    y1, sr1 = librosa.load(file1, sr=None)
    y2, sr2 = librosa.load(file2, sr=None)

    # Normalisation optionnelle
    if normalize:
        y1 = y1 / (np.max(np.abs(y1)) + 1e-9)
        y2 = y2 / (np.max(np.abs(y2)) + 1e-9)

    # --- 2. Alignement sur l'attaque ---
    def align_on_onset(y, sr):
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        if len(onset_frames) > 0:
            onset_sample = librosa.frames_to_samples(onset_frames[0])
            return y[onset_sample:]
        return y

    y1 = align_on_onset(y1, sr1)
    y2 = align_on_onset(y2, sr2)

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

    # --- PARTIE B : SIMILARITÉ D’ENVELOPPE RMS ---

    hop = 512
    env1 = librosa.feature.rms(y=y1, frame_length=1024, hop_length=hop)[0]
    env2 = librosa.feature.rms(y=y2, frame_length=1024, hop_length=hop)[0]

    if len(env1) > 1 and len(env2) > 1:
        score_env = np.corrcoef(env1, env2)[0, 1]
    else:
        score_env = 0

    score_env = float(np.clip(score_env, -1, 1))
    if np.isnan(score_env) or score_env < 0:
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
