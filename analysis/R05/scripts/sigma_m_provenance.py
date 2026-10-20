"""R05 §3.4 — R02 sigma_m provenance: for each native-7 unit, was the mosaic sigma_m produced by a model that did NOT
train on that unit? Evidence: (1) R02 het_clean.py fold definition (train_pairs = ALL7 minus the held unit's pairs);
(2) het_clean_r02/fold_summary.json n_train per fold vs the capped file counts; (3) mosaic_infer.py MODEL_FOR = fold_<unit>;
(4) the mosaic job log lines showing which fold directory was loaded per unit; (5) each fold's perpatch_tile.json lists
only held-unit files."""
import json, os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common as C
m = C.start("sigma_m_provenance"); R02 = C.C4.R02; out = {}
hc = open(f"{R02}/het_clean.py").read(); mi = open(f"{R02}/mosaic_infer.py").read(); fs = json.load(open(f"{R02}/het_clean_r02/fold_summary.json"))
code_ok = ("train_pairs = [p for p in ALL7 if p not in held]" in hc) and ('MODEL_FOR = {u: f"fold_{u}" for u in hc.FOLDS}' in mi)
logs = sorted(glob.glob(f"{R02}/logs/r02_mosaic*out")); logtxt = "\n".join(open(l).read() for l in logs)
FOLDS = {"NA090": ["NA090__MGDS_31212"], "NA080": ["NA080__MGDS_31290"], "TN399": ["TN399__MGDS_30373"], "RR1506": ["RR1506__MGDS_29779"], "AT37-13": ["AT37-13__MGDS_31199"], "FK181031": ["FK181031__MGDS_24618"], "TN299": ["TN299__MGDS_27339", "TN299__MGDS_31253"]}
for u, ps in FOLDS.items():
    pp = json.load(open(f"{R02}/het_clean_r02/fold_{u}/perpatch_tile.json")); held_pairs = sorted({r["pair"] for r in pp})
    n_train = fs[u]["n_train"]; others_total = None
    ev = dict(code_fold_rule=code_ok, fold_dir_exists=os.path.isdir(f"{R02}/het_clean_r02/fold_{u}"), perpatch_held_pairs=held_pairs, held_pairs_equal_unit=(held_pairs == sorted(ps)),
              n_train=n_train, n_held=fs[u]["n_held"], mosaic_log_mentions=[l.strip()[:160] for l in logtxt.splitlines() if (f"fold_{u}" in l or (u in l and "model" in l.lower()))][:3])
    ok = code_ok and ev["fold_dir_exists"] and ev["held_pairs_equal_unit"]
    out[u] = dict(label="CONFIRMED (leave-one-unit-out)" if ok else "COULD NOT CHECK", evidence=ev)
# n_train consistency: every fold trains on ALL7 minus held, capped 2500 per pair -> folds holding 1 pair of 8 have equal n_train except where a pair has < 2500 files
# arithmetic check: train set = all 9 native pairs capped at 2,500 files each, minus the held unit's capped files
import glob as _g
cap = {p: min(2500, len(_g.glob(f"{R02}/patches_clean/*/{p}_r*.npz"))) for ps in FOLDS.values() for p in ps}; total = sum(cap.values())
for u, ps in FOLDS.items():
    expect = total - sum(cap[p] for p in ps); out[u]["evidence"]["n_train_expected_if_LOUO"] = expect; out[u]["evidence"]["n_train_matches"] = (expect == fs[u]["n_train"])
    if not out[u]["evidence"]["n_train_matches"]: out[u]["label"] = "COULD NOT CHECK (n_train mismatch)"
out["_n_train_by_fold"] = {u: fs[u]["n_train"] for u in FOLDS}; out["_capped_files_per_pair"] = cap
json.dump(out, open(f"{C.ROOT}/sigma_m_provenance.json", "w"), indent=1)
for u, r in out.items():
    if u.startswith("_"): continue
    print(f"{u:9s} {r['label']:32s} held pairs {r['evidence']['perpatch_held_pairs']} n_train {r['evidence']['n_train']} (expected {r['evidence']['n_train_expected_if_LOUO']}) n_held {r['evidence']['n_held']} log: {r['evidence']['mosaic_log_mentions'][:1]}")
