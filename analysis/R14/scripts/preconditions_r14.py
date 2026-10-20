"""R14 §2 preconditions (read only; nothing sealed is opened):
  1. the five Phase 4 scripts equal the v2.1 B6 table and are clean; `git status --porcelain phase4/` recorded (the 'before' snapshot);
  2. the R09 guard self-test 702 / 702 (run by r14_common.start through phase4_common.start);
  3. the development pair set of the R10 C1 calibration: the 15 pairs of the ten real-relief units with each pair its own unit,
     the 14 in R10's C1 set (R10b per_pair_D1.csv, OAK-verified), with unit, hr_res, lr_native, k, D1 and native cells across;
  4. the GBT_ship leave-one-unit-out fold models: search R04 / R05 / R09 scratch and OAK for persisted fold models; the per-fold
     configurations the frozen bar chose (R05 chosen_configs_tile.json) against cfg 18 (the all-16 artifact's configuration).
Writes R14/tables/preconditions.json (+ .md through report_tables_r14.py)."""
import glob, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_common as R
P4, C4, R8 = R.P4, R.C4, R.R8

def git(*a): return subprocess.run(["git", *a], cwd=R.REPO, capture_output=True, text=True).stdout.strip()
def rel(p): return os.path.relpath(p, R.REPO)

if __name__ == "__main__":
    m, rec = R.start("preconditions_r14.py"); out = dict(git_head=git("rev-parse", "HEAD"), guard=dict(blocked=rec["guard_probes_blocked"], probes=rec["guard_probes"]))
    # 1. scripts vs the v2.1 B6 table; git status of phase4/
    stated = P4.parse_prereg_hashes(open(P4.PREREG_V2).read()); sc = {}
    for s in P4.SCRIPTS:
        p = os.path.join(P4.HERE, s); h = P4.sha256_file(p); sc[s] = dict(on_disk=h, b6=stated.get(s), equal=(h == stated.get(s)))
    out["1_scripts"] = sc; out["1_phase4_git_status_before"] = git("status", "--porcelain", "--", rel(os.path.dirname(P4.HERE))); out["1_prereg_v2_1_sha256"] = P4.sha256_file(P4.PREREG_V2)
    out["1_passed"] = bool(all(v["equal"] for v in sc.values()) and out["1_phase4_git_status_before"] == "")
    # 2. guard
    out["2_passed"] = bool(rec["guard_probes_blocked"] == rec["guard_probes"] == 702)
    # 3. the pair set
    SI, PP, cmap, stats = R.load_r10b(); tiles = R8.load_tiles(); rows = []
    for pid in R.PAIRS15:
        g = R.pair_geometry(tiles, pid); o = PP.loc[pid]; plan, skipped = R.level_plan(g["hr_res"], g["lr_native"])
        rows.append(dict(pair=pid, unit=C4.UNIT_OF[pid], **g, cells_r10b=float(o["cells_across"]), view_r10=o["view"], status_r10=o["status"], in_c1_set=bool(str(o["status"]).startswith("in C1")), rho_s_lr_r10=float(o["rho_s_lr"]), rho_GBT_r10=float(o["rho_GBT_ship"]),
                         levels=[c for c in R.LEVELS if c in plan], lr_target={str(c): v for c, v in plan.items()}, skipped=[c for c in R.LEVELS if c in skipped]))
    c1 = [r["pair"] for r in rows if r["in_c1_set"]]; out["3_pairs"] = rows; out["3_c1_set"] = c1; out["3_n_c1"] = len(c1)
    out["3_passed"] = bool(len(c1) == 14 and set(c1) == set(R.PAIRS15) - {"AT42-06__MGDS_5174"} and sum(bool(PP.loc[p, "c1_condition"]) for p in c1) == 9)
    out["3_n_pass_r10"] = int(sum(bool(PP.loc[p, "c1_condition"]) for p in c1))
    # 4. fold models
    found = []
    for root in (f"{C4.ROOT}", "/scratch/users/scdobbs/grl_review/R05", "/scratch/users/scdobbs/grl_review/R09", f"{R8.OAK_ARCHIVE}/R04", f"{R8.OAK_ARCHIVE}/R05", f"{R8.OAK_ARCHIVE}/R09"):
        for ext in ("joblib", "pkl", "pickle", "skops"):
            found += glob.glob(f"{root}/**/*.{ext}", recursive=True)
    out["4_model_files_found"] = {p: P4.sha256_file(p) for p in sorted(found)}
    chosen = json.load(open(R8.CHOSEN_TILE))["tile/A_fine/GBT_ship"]; r09 = json.load(open(f"{R.R9.OUT_ART}/gbt_ship_all16_record.json"))
    out["4_frozen_bar_fold_cfgs"] = {u: dict(cfg_idx=v["cfg_idx"], cfg=v["cfg"], inner_score=v["inner_score"]) for u, v in chosen.items()}
    out["4_cfg18"] = dict(cfg_idx=r09["cfg_idx"], cfg=r09["config"], artifact=r09["artifact"], sha256=r09["sha256"], note="R09 all-16 artifact (trained on all 16 development units; NOT used here, directive §4.3)")
    out["4_n_folds_using_cfg18"] = int(sum(v["cfg_idx"] == R.CFG18 for v in chosen.values()))
    out["4_fold_models_persisted"] = False; out["4_note"] = "No leave-one-unit-out fold model was persisted by R04, R05 or R09 (R05 persisted predictions and chosen configurations only; R09 persisted the all-16 model only); the frozen bar's outer folds used per-fold inner-tuned configurations, none of them cfg 18 -> §4.3 refit, see gbt_folds_r14.py"
    out["all_passed"] = bool(out["1_passed"] and out["2_passed"] and out["3_passed"])
    R.dump(out, f"{R.TABLES}/preconditions.json"); print(json.dumps({k: v for k, v in out.items() if k not in ("3_pairs", "4_model_files_found")}, indent=1, default=str)); print("model files found:", len(found))
