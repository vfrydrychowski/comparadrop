import os
import numpy as np
from itertools import combinations
import json
from datetime import datetime
import argparse
from audio_processing import get_hybrid_score

# --------------------------------------------------------------
#       COMPARAISON DE TOUS LES FICHIERS D'UN DOSSIER
# --------------------------------------------------------------

def compare_folder(folder_path, weights=(0.6, 0.4)):

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
        )
        results.append((f1, f2, score["total"]))

    # Statistiques
    if not results:
        return {
            "min": None, "max": None, "mean": 0.0, "all_pairs": []
        }

    scores = [r[2] for r in results]
    min_pair = results[int(np.argmin(scores))]
    max_pair = results[int(np.argmax(scores))]
    avg_score = np.mean(scores)

    # Use os.path.basename to remove folder path from filenames in the output
    summary = {
        "min": {"files": (os.path.basename(min_pair[0]), os.path.basename(min_pair[1])), "score": min_pair[2]},
        "max": {"files": (os.path.basename(max_pair[0]), os.path.basename(max_pair[1])), "score": max_pair[2]},
        "mean": float(avg_score),
        "all_pairs": [(os.path.basename(r[0]), os.path.basename(r[1]), r[2]) for r in results]
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
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Directory to save the JSON results file. Defaults to the current directory."
    )
    args = parser.parse_args()

    # --- 1. Run comparison ---
    summary = compare_folder(args.folder)

    # --- 2. Save results to a timestamped JSON file ---
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_results_{timestamp}.json"
    output_path = os.path.join(args.output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Comparison summary saved to: {output_path}")
