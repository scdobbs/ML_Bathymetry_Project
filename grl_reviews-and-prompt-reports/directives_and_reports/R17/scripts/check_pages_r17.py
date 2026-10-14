"""R17 §2.4–2.5 checks on Phase 5 figure set A (grl_review_archive/phase5/figures), WITHOUT opening the key, the work pages or
pages_record.json:
  - page count of figset_A_blind.pdf (pypdf) and a text extraction per page (expected: no text objects; the pages are rasterised PNGs);
  - the Q-codes, taken from the set-A page FILE NAMES under figures/work/pages/A_Q*.png (listing only; the files are not opened) and
    checked against Q01..Qn for n = page count;
  - the page-text rule as coded (fields_r08.py: suptitle = code, axes off, scalebar label '500 m'), restated by reading the source lines;
  - writes R17/phase5_calls_TEMPLATE.csv (one row per code, pre-registered columns, value columns empty) and records sha256 of the PDF and
    of the template.
No image is rendered or described."""
import csv, glob, hashlib, json, os, re, sys
from pypdf import PdfReader
HERE = os.path.dirname(os.path.abspath(__file__)); DR = os.path.dirname(os.path.dirname(HERE)); R17 = os.path.dirname(HERE)
ROOT = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive/phase5"; PDF = f"{ROOT}/figures/figset_A_blind.pdf"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
if __name__ == "__main__":
    rd = PdfReader(PDF); n = len(rd.pages); texts = [(pg.extract_text() or "").strip() for pg in rd.pages]
    codes = sorted(re.match(r"A_(Q\d{2})\.png$", os.path.basename(p)).group(1) for p in glob.glob(f"{ROOT}/figures/work/pages/A_Q*.png"))
    expected = [f"Q{i:02d}" for i in range(1, n + 1)]
    src = open(f"{DR}/R08/scripts/fields_r08.py").read().splitlines(); rule = {i + 1: l.strip()[:150] for i, l in enumerate(src) if ("suptitle(D[\"code\"][pid]" in l) or ("f\"{int(R8.SCALEBAR_M)} m\"" in l) or ("set_axis_off" in l and "scalebar" in l)}
    tpl = f"{R17}/phase5_calls_TEMPLATE.csv"
    with open(tpl, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["code", "relief", "criteria", "dominant", "ab_whole_survey", "note"]); [w.writerow([c, "", "", "", "", ""]) for c in expected]
    out = dict(pdf=PDF, pdf_sha256=sha(PDF), n_pages=n, text_objects_per_page=[len(t) for t in texts], no_text_objects=bool(all(len(t) == 0 for t in texts)), codes_from_work_page_filenames=codes, codes_expected=expected, codes_match=bool(codes == expected),
               key_exists=os.path.exists(f"{ROOT}/figures/key/figset_A_key.json"), key_not_in_pdf_text=bool(not any(re.search(r"MGDS|__|lu\d\d|nu\d\d|p5u\d\d|Escanaba|KN161|KN173|MV1003|AT26", t) for t in texts)), page_text_rule_as_coded=rule, template=tpl, template_sha256=sha(tpl), template_rows=len(expected),
               passed=bool(n == len(expected) and all(len(t) == 0 for t in texts) and codes == expected))
    os.makedirs(f"{R17}/tables", exist_ok=True); json.dump(out, open(f"{R17}/tables/pages_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
