import glob
import json
import os
import random
import time
from itertools import product
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from stair_counter_visual import count_stairs_approachB_visual


def ground_truth_from_json(json_path: str) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        return f.read().count('"label"')


def list_image_json_pairs(dir_path: str) -> List[Tuple[str, str]]:
    imgs = sorted(glob.glob(os.path.join(dir_path, "*.jpg")))
    pairs: List[Tuple[str, str]] = []
    for img in imgs:
        json_path = os.path.splitext(img)[0] + ".json"
        if os.path.exists(json_path):
            pairs.append((img, json_path))
        else:
            raise FileNotFoundError(f"Pas de JSON pour {img}")
    return pairs


def evaluate_dataset(dir_path: str, detection_fn, **params) -> pd.DataFrame:
    """Construit un DataFrame [image, pred, gt, abs_err] pour *dir_path*."""
    recs = []
    for img_path, json_path in list_image_json_pairs(dir_path):
        pred = detection_fn(img_path, **params)
        gt = ground_truth_from_json(json_path)
        recs.append({
            "image": os.path.basename(img_path),
            "pred": pred,
            "gt": gt,
            "abs_err": abs(pred - gt),
        })
    return pd.DataFrame.from_records(recs)


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    mae = df["abs_err"].mean()
    acc = (df["abs_err"] == 0).mean() * 100
    total_gt = df["gt"].sum()
    total_pred = df["pred"].sum()
    rate_simple = total_pred / total_gt * 100 if total_gt else 0
    correct_detected = np.minimum(df["pred"], df["gt"]).sum()
    rate_corrects = correct_detected / total_gt * 100 if total_gt else 0
    return {
        "MAE": mae,
        "Accuracy": acc,
        "Rate_simple": rate_simple,
        "Rate_corrects": rate_corrects,
    }


def pretty_print(tag: str, metrics: Dict[str, float]):
    print(f"\n{tag}")
    print(f"MAE                   : {metrics['MAE']:.2f} marches")
    print(f"Accuracy              : {metrics['Accuracy']:.1f}% images exactes")
    print(f"- Détection simple    : {metrics['Rate_simple']:.1f}%")
    print(f"- Détection corrects : {metrics['Rate_corrects']:.1f}%")


BASE_PARAMS = {
    "blur_ksize": (5, 5),
    "canny_thresh1": 60,
    "canny_thresh2": 150,
    "close_kernel_size": (3, 3),
    "hough_threshold": 40,
    "min_line_length": 80,
    "max_line_gap": 15,
    "angle_tolerance": 6,
    "y_group_distance": 12,
    "discard_small_groups": True,
    "group_min_size": 2,
    "apply_clahe": True,
    "length_ratio_threshold": 0.4,
}


SEARCH_SPACE = {
    "blur_ksize": [(3, 3), (5, 5), (7, 7)],
    "canny_thresh1": [40, 50, 60, 70],
    "canny_thresh2": [100, 130, 160, 190],
    "min_line_length": [50, 70, 90],
    "y_group_distance": [8, 10, 12, 15],
    "apply_clahe": [True, False],
}


def sample_params(k: int) -> List[Dict[str, Any]]:
    keys = list(SEARCH_SPACE.keys())
    return [
        {key: random.choice(SEARCH_SPACE[key]) for key in keys}
        for _ in range(k)
    ]


def random_search(n_trials: int, seed: int = 42, scoring: str = "10*acc-mae") -> Dict[str, Any]:
    random.seed(seed)
    best: Dict[str, Any] | None = None
    logs: List[Dict[str, Any]] = []
    t0 = time.time()

    def score_fn(mae: float, acc: float) -> float:
        if scoring == "acc-mae":
            return acc - mae
        return 10 * acc - mae

    for i, params in enumerate(sample_params(n_trials), 1):
        df = evaluate_dataset("train_val", count_stairs_approachB_visual, **params)
        m = compute_metrics(df)
        score = score_fn(m["MAE"], m["Accuracy"])
        record = {**params, **m, "score": score}
        logs.append(record)
        if best is None or score > best["score"]:
            best = record
            print(
                f"[{i}/{n_trials}] new best → acc={m['Accuracy']:.2f}%, mae={m['MAE']:.2f}, params={json.dumps(params)}"
            )

    pd.DataFrame(logs).to_csv("random_search_log.csv", index=False)
    print(f"random_search_log.csv sauvegardé ( {len(logs)} essais, {time.time()-t0:.1f}s )")
    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Évalue ou cherche automatiquement les meilleurs paramètres.")
    sub = parser.add_subparsers(dest="cmd", help="Sous‑commandes")

    p_eval = sub.add_parser("eval", help="Évaluer train_val et/ou test avec BASE_PARAMS ou un fichier JSON")
    p_eval.add_argument("--split", default="train_val,test", help="train_val,test,all")
    p_eval.add_argument("--params-json", help="Chemin d'un fichier JSON contenant un dictionnaire de paramètres")

    p_rand = sub.add_parser("random", help="Random‑search sur train_val")
    p_rand.add_argument("--trials", type=int, default=200, help="Nombre d'essais (défaut 200)")
    p_rand.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "random":
        best = random_search(args.trials, seed=args.seed)
        print("\nBEST FOUND")
        print(json.dumps(best, indent=2, ensure_ascii=False))

    else:
        params = BASE_PARAMS
        if args.params_json:
            with open(args.params_json, "r", encoding="utf-8") as f:
                params = json.load(f)
        splits = [s.strip() for s in args.split.split(",") if s.strip()]
        for sp in splits:
            df = evaluate_dataset(sp, count_stairs_approachB_visual, **params)
            metrics = compute_metrics(df)
            pretty_print(sp.upper(), metrics)
            df.to_csv(f"{sp}_results.csv", index=False)
            print(f"Résultats sauvegardés dans {sp}_results.csv")
