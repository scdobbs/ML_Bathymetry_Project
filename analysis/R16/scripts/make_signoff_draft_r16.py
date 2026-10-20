"""R16 Part E2 — write R16/SIGNOFF_phase5_prereg_v3_DRAFT.md on the v2.1 sign-off pattern. Its first 64-hex string is the v3 sha256
(checked with the lock's own phase5_common.read_signoff == phase4_common.read_signoff). Signature blank. The draft keeps the
_DRAFT name so that the phase5 lock (which reads R16/SIGNOFF_phase5_prereg_v3.md) cannot pass unsigned."""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, f"{DR}/phase5/scripts")
V3 = f"{DR}/R16/PREREG_phase5_v3.md"; OUT = f"{DR}/R16/SIGNOFF_phase5_prereg_v3_DRAFT.md"; P5D = f"{DR}/phase5/scripts"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
TEMPLATE = """# SIGNOFF_phase5_prereg_v3 (DRAFT, unsigned)

The CNN repo's Phase 5 locks read the signed file. `phase5_common.read_signoff` (the imported `phase4_common.read_signoff`; constant `SIGNOFF = R16/SIGNOFF_phase5_prereg_v3.md`) takes the first sha256 it holds and requires it to equal the pre-registration's. The acquisition repo's `sealed_step1_p5.py` lock 1 reads the sign-off and the pre-registration sha256 too. This draft is kept under the `_DRAFT` name, so no lock can pass before it is signed. To sign: fill the signature and date, then save it as `R16/SIGNOFF_phase5_prereg_v3.md` and commit it. After that, tag `phase5-prereg-v3` (R17's first step).

| field | value |
|---|---|
| pre-registration sha256 | `{v3}` |
| pre-registration | `directives_and_reports/R16/PREREG_phase5_v3.md` (version 3, rendered from `R12c/PREREG_phase4_v2_1.md`, sha256 `d6af268b…8334`, tag `phase4-prereg-v2.1`, which stays untouched) |
| design spec | `directives_and_reports/phase_5/PHASE5_design_spec_2026-10-13.md` (sha256 `{spec}`; ruled 2026-10-13) |
| Phase 5 scripts (v3 B6) | {rows5} |
| Phase 4 scripts imported unchanged (v2.1 B6) | `phase4_common.py` `99407ec6…`, `phase4_build.py` `98370164…`, `phase4_figures.py` `44e87590…`, `phase4_predict.py` `7267a9c8…`, `phase4_score.py` `9aefb6ff…` |
| CNN tag | `phase5-prereg-v3` (to be set on the commit holding v3 and the signed sign-off; R17) |
| acquisition freeze tag | `phase5-acq-freeze` |
| acquisition `PHASE5/freeze_record_phase5.json` sha256 | `6769583198da63d874472ea5a2ca62cb26c2f9f55003eb6e97838bfebab1d9d1` |
| acquisition `src/phase5/sealed_step1_p5.py` sha256 | `9f143e0c3f000163d9da48001cbd7578805c8c1042141f9994a972df73b1b582` (code commit `4d93c8a`) |
| acquisition supplement | `manifest/pairs_phase5_sealed.parquet`, not yet written (ACQ-R10); its sha256 goes into `CHANGE_LEDGER.md` before R17 |
| signed by | |
| date | |

## Exposure statement (pre-registration section "Prior exposure")

Steve has seen no bathymetry of the eight sites. That includes the June QC figure for Bermuda and the La Jolla Zephyr survey (stated 2026-10-13). The only candidate numbers seen are the ACQ-R08 and ACQ-R08b metadata, listed in REPORT_ACQ-R08b §5:

- per grid: file names, sizes and sha256, grid size, cell size, CRS, extent, nodata and dtype;
- per candidate: the valid area, the footprint polygon and its area, one median depth rounded to 100 m, and the ship cruise, sonar, year and swath file count / size;
- coverage: navigation-track counts, the half-swath buffer, the coverage area and fraction;
- the estimates `lr_est_m`, `cells_across_est`, `ratio_est` and `tiles_est`;
- `same_site_as`, the nearest corpus footprint and status.

Per site:

| site | HR id | HR cell (m) | valid km² | depth, nearest 100 m | ship | lr_est (m) | cells across (est.) | tiles (est.) |
|---|---|---|---|---|---|---|---|---|
| Galápagos Spreading Center | MGDS:18210 | 1.0025 | 75.786 | 1700 | TN188 | 29.67 | 8.65 | 3917 |
| Escanaba Trough | MGDS:30368 | 1.1477 | 84.435 | 3300 | Escanaba | 57.6 | 5.1 | 3918 |
| Monterey Canyon | MGDS:24756 | 1.0 | 27.021 | 1400 | Monterey | 24.44 | 10.47 | 1611 |
| Mid-Atlantic Ridge 16.5°N | MGDS:22436 | 2.0033 | 90.606 | 3400 | KN161L04 | 118.69 | 4.32 | 1351 |
| La Jolla Canyon | MGDS:21415 | 1.0006 | 24.373 | 600 | MV1209 | 10.47 | 24.47 | 1211 |
| Bermuda | MGDS:31675 | 1.0002 | 9.159 | 1600 | KN173L02 | 55.86 | 4.58 | 527 |
| Chile margin | MGDS:31860 | 0.9988 | 7.149 | 2900 | MV1003 | 50.62 | 5.05 | 450 |
| Dorado Outcrop | MGDS:31954 | 1.0032 | 5.584 | 3100 | AT26-24 | 54.11 | 4.75 | 331 |

The ACQ-R08b metadata also cover MGDS:31811, the Galápagos 2 m product that is not used (depth 2700, nearest 100 m). The executor that ran ACQ-R08b saw only those numbers. The R16 executor opened no candidate file. It read REPORT_ACQ-R08b §5 to write this list.

The blind artifact call of B3 step 3 is made against the R08 §3.1 criteria with this exposure stated.

[ ] confirmed as written      [ ] corrected as follows:

## Procedure note (carried from v2.1)

The amplitude statistic and the figure key exist after B3 step 2. Steve does not see either until his blind calls are committed. The step-2 report carries no amplitude and no key.

## Declaration

I have read `PREREG_phase5_v3.md` at the sha256 above and sign it as the pre-registration for Phase 5. Phase 5 runs in the order of its B3 and reports every result, whatever it is.

Signature: ______________________  Date: __________
"""
if __name__ == "__main__":
    import phase5_common as P5
    v3 = sha(V3); rows5 = ", ".join(f"`{s}` `{sha(os.path.join(P5D, s))[:8]}…`" for s in P5.SCRIPTS)
    open(OUT, "w").write(TEMPLATE.format(v3=v3, spec=sha(f"{DR}/phase_5/PHASE5_design_spec_2026-10-13.md"), rows5=rows5))
    first = P5.read_signoff(OUT); ok = first == v3
    rec = dict(draft=OUT, draft_sha256=sha(OUT), v3_sha256=v3, read_signoff_first_sha256=first, read_signoff_equals_v3=ok); print(json.dumps(rec, indent=1)); assert ok
    json.dump(rec, open(f"{DR}/R16/tables/signoff_draft_check.json", "w"), indent=1)
