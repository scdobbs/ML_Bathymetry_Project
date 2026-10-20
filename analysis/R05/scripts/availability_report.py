"""R05 §2.3 — availability report with the consumer-side override: verifies on features_v1.parquet (R04, not recomputed)
that the overridden feature is already NaN for the pair, and records per-pair availability of the contract features."""
import json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import r05_common as C, features_v1_1 as F
m = C.start("availability_report"); f = pd.read_parquet(f"{C.R04}/features/features_v1.parquet"); av = json.load(open(f"{C.R04}/features/features_v1_availability.json"))
out = {}
for pid, a in av.items():
    s = f[f.pair == pid]; ov = {prod: reason for (p, prod), reason in F.OVERRIDES.items() if p == pid}
    cols = {k: [f"{k}_{w}" for w in F.WINDOWS_PX] for k in F.CONTRACT_KEYS}
    defined = {k: float(np.mean([np.isfinite(s[c]).mean() for c in cc])) for k, cc in cols.items()}
    out[pid] = dict(unit=a["unit"], usable=a["usable"], n_rows=a["n_rows"], override=ov, frac_defined=defined,
                    override_verified={prod: bool(all((~np.isfinite(s[c])).all() for k, p2 in F.PRODUCT_OF_KEY.items() if p2 == prod for c in cols[k])) for prod in ov})
json.dump(out, open(f"{C.ROOT}/features_v1_1_availability.json", "w"), indent=1)
for pid, o in out.items():
    if o["override"]: print(pid, o["override"], "verified all-NaN in features_v1.parquet:", o["override_verified"], "frac_defined:", {k: round(v, 2) for k, v in o["frac_defined"].items()})
print("usable pairs:", sum(o["usable"] for o in out.values()), "/", len(out))
