import json, os, sys
from pathlib import Path

# -- chemins ---------------------------------------------------------------
SRC = Path(__file__).parent
PROJ = SRC.parent
sys.path.append(str(SRC))          # pour importer les detecteurs

from evaluate import evaluate_dataset, compute_metrics   # déjà fourni

# -- import des deux versions ---------------------------------------------
from stair_counter_old     import count_stairs_approachB_visual as detect_old
from stair_counter_visual  import count_stairs_approachB_visual as detect_new

# -- fichiers params -------------------------------------------------------
with open(PROJ / "params" / "baseline.json") as f:
    P_BASE = json.load(f)
with open(PROJ / "params" / "optimal.json") as f:
    P_OPT  = json.load(f)

SPLITS = ["train_val", "test"]

def bench(label, fn, split, params):
    df = evaluate_dataset(PROJ / "data" / split, fn, **params)
    m  = compute_metrics(df)
    print(f"{label:6} | {split:9} | MAE={m['MAE']:.2f} | Acc={m['Accuracy']:.1f}% | simple={m['Rate_simple']:.1f}% | corrects={m['Rate_corrects']:.1f}%")
    df.to_csv(f"{label.lower()}_{split}_results.csv", index=False)

for split in SPLITS:
    bench("OLD",  detect_old, split, P_BASE)
    bench("NEW",  detect_new, split, P_OPT)
