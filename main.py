import os
import numpy as np
from itertools import combinations
import argparse
from audio_processing import get_hybrid_score

# --------------------------------------------------------------
#       COMPARAISON DE TOUS LES FICHIERS D'UN DOSSIER
# --------------------------------------------------------------

def compare_folder(folder_path, weights=(0.6, 0.4), normalize=False):

    files = [os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".wav", ".flac", ".aiff", ".aif", ".mp3"))]

    if len(files) < 2:
        raise ValueError("Il faut au moins deux fichiers audio dans le dossier.")

    results = []
    
    for f1, f2 in combinations(files, 2):
        score = get_hybrid_score(
            f1, f2,
            weights=weights,
            normalize=normalize,
            verbose=False
        )
        results.append((f1, f2, score["total"]))

    # Statistiques
    scores = [r[2] for r in results]
    min_pair = results[int(np.argmin(scores))]
    max_pair = results[int(np.argmax(scores))]
    avg_score = np.mean(scores)

    summary = {
        "min": {"files": (min_pair[0], min_pair[1]), "score": min_pair[2]},
        "max": {"files": (max_pair[0], max_pair[1]), "score": max_pair[2]},
        "mean": float(avg_score),
        "all_pairs": results
    }

    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare audio files in a folder.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="samples/",
        help="Path to the folder containing audio files. Defaults to 'samples/'."
    )
    args = parser.parse_args()

    # Exemple d’utilisation :
    summary = compare_folder(args.folder, normalize=False)
    print(summary)
