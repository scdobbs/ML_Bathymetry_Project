"""R23 common style for Figures 1-3 of the GRL revision (directive R23 §0). Imported by fig1.py, fig2.py, fig3.py.

AGU/GRL: 170 mm two-column width; Arial if fc-list finds it, else Liberation Sans (Arial-metric), embedded; 7 pt text,
8 pt bold lowercase panel letters; 0.5 pt axes; no titles, no legends unless unavoidable, no frames; Okabe-Ito categorical
colours, Crameri sequential/diverging maps; Illustrator-editable output (TrueType fonts, text kept as text in SVG,
rasters only for image layers at 600 dpi, no alpha on vector elements)."""
from __future__ import annotations
import hashlib, json, os, re, subprocess
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

MM = 1.0 / 25.4
WIDTH_MM = 170.0
FS = 7.0             # tick labels, axis labels, colorbar labels, direct labels
FS_PANEL = 8.0       # panel letters (bold, lowercase)
MIN_FS = 6.0         # GRL minimum, incl. sub/superscripts
LW_AX = 0.5; LW_DATA = 0.8; LW_THIN = 0.5; MS = 3.5
OKABE = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9", green="#009E73", yellow="#F0E442",
             blue="#0072B2", vermillion="#D55E00", purple="#CC79A7")
GRAY_ZERO = "#9A9A9A"; GRAY_LIGHT = "#B8B8B8"; GRAY_BAND = "#E4E4E4"; GRAY_LINE = "#808080"

REPO = "/home/users/scdobbs/GAN_Bathymetry_Project"
OUT = f"{REPO}/analysis/R23/figures"
AR = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy/grl_review_archive"
OAKD = "/oak/stanford/groups/hilley/auv_ship_colocated_bathy"
ACQ = "/home/users/scdobbs/co-located-auv-ship-based-multibeam-acquisition"
TMP = "/scratch/users/scdobbs/tmp_r23"
USER_FONT_DIR = os.path.expanduser("~/.local/share/fonts")

FONT = {"name": None, "files": [], "source": None}


def _fc_list():
    try:
        return subprocess.run(["fc-list"], capture_output=True, text=True, timeout=120).stdout.splitlines()
    except Exception:
        return []


def pick_font():
    """Arial if fc-list finds it; otherwise Liberation Sans. Registers the ttf files with matplotlib and returns the name."""
    lines = _fc_list(); found = {}
    for l in lines:
        parts = l.split(":")
        if len(parts) < 2: continue
        path, fams = parts[0].strip(), [f.strip() for f in parts[1].split(",")]
        for fam in fams: found.setdefault(fam, []).append(path)
    for name in ("Arial", "Liberation Sans"):
        files = [p for p in found.get(name, []) if p.lower().endswith(".ttf")]
        if files:
            FONT.update(name=name, files=sorted(set(files)), source="fc-list"); break
    if FONT["name"] is None:
        files = sorted(f for f in os.listdir(USER_FONT_DIR) if f.startswith("LiberationSans") and f.endswith(".ttf")) if os.path.isdir(USER_FONT_DIR) else []
        if not files: raise SystemExit("neither Arial nor Liberation Sans available; install Liberation Sans in ~/.local/share/fonts")
        FONT.update(name="Liberation Sans", files=[os.path.join(USER_FONT_DIR, f) for f in files], source="~/.local/share/fonts")
    for f in FONT["files"]: fm.fontManager.addfont(f)
    return FONT["name"]


def apply():
    name = pick_font()
    import matplotlib._mathtext as _mt
    assert hasattr(_mt, "SHRINK_FACTOR"), "matplotlib._mathtext.SHRINK_FACTOR not found"
    _mt.SHRINK_FACTOR = MIN_FS / FS      # sub/superscripts at 7 pt base render at 6 pt (GRL minimum), not the default 0.7 x 7 = 4.9 pt
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif", "font.sans-serif": [name],
        "mathtext.fontset": "custom", "mathtext.rm": name, "mathtext.it": f"{name}:italic", "mathtext.bf": f"{name}:bold",
        "mathtext.sf": name, "mathtext.cal": name, "mathtext.tt": name, "mathtext.default": "regular", "mathtext.fallback": None,
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.linewidth": LW_AX, "xtick.major.width": LW_AX, "ytick.major.width": LW_AX, "xtick.minor.width": LW_AX, "ytick.minor.width": LW_AX,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8, "axes.labelpad": 2.0,
        "lines.linewidth": LW_DATA, "lines.markersize": MS, "lines.markeredgewidth": 0.5,
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
        "axes.unicode_minus": True, "savefig.transparent": False, "figure.dpi": 100, "savefig.dpi": 600,
        "image.interpolation": "antialiased", "axes.grid": False, "axes.titlesize": FS, "figure.titlesize": FS,
    })
    return name


def fig_new(w_mm: float = WIDTH_MM, h_mm: float = 100.0):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def ax_mm(fig, x_mm, y_top_mm, w_mm, h_mm, **kw):
    """Axes placed in millimetres from the figure's top-left corner."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x_mm / W, 1.0 - (y_top_mm + h_mm) / H, w_mm / W, h_mm / H], **kw)


def style_axes(ax, spines=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in spines); ax.spines[s].set_linewidth(LW_AX)
    ax.tick_params(width=LW_AX, labelsize=FS, length=2.0, pad=1.8)
    ax.grid(False)
    return ax


def panel_letter(fig, ax, letter, dx_mm=-5.0, dy_mm=1.0, gid=None):
    """8 pt bold lowercase letter at the top-left corner, outside the axes (offset in mm)."""
    W, H = fig.get_size_inches() / MM
    b = ax.get_position()
    t = fig.text(b.x0 + dx_mm / W, b.y1 + dy_mm / H, letter, fontsize=FS_PANEL, fontweight="bold", ha="left", va="bottom")
    t.set_gid(gid or f"panel_{letter}")
    return t


def zero_line(ax, axis="x", gid=None):
    f = ax.axvline if axis == "x" else ax.axhline
    l = f(0.0, color=GRAY_ZERO, linewidth=LW_THIN, zorder=0.5); l.set_gid(gid or f"zero_{axis}"); return l


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def save_all(fig, n: int, out=OUT):
    os.makedirs(out, exist_ok=True); base = f"{out}/Dobbs_Figure{n}"; paths = {}
    fig.savefig(f"{base}.pdf", dpi=600, format="pdf"); paths["pdf"] = f"{base}.pdf"
    fig.savefig(f"{base}.eps", dpi=600, format="eps"); paths["eps"] = f"{base}.eps"
    fig.savefig(f"{base}.svg", dpi=600, format="svg"); paths["svg"] = f"{base}.svg"
    fig.savefig(f"{base}_preview.png", dpi=300, format="png"); paths["png"] = f"{base}_preview.png"
    return paths


def check_pdf(pdf_path, font_name):
    """pdffonts: every font embedded and Arial / Liberation Sans; pdfinfo: page width 170 ± 0.5 mm."""
    fonts = subprocess.run(["pdffonts", pdf_path], capture_output=True, text=True, check=True).stdout
    rows = [l for l in fonts.splitlines()[2:] if l.strip()]
    ok_fonts, bad = True, []
    key = font_name.replace(" ", "")
    for l in rows:
        # columns: name type encoding emb sub uni object-id ("18 0"); type may contain spaces -> parse from the right
        tok = l.split(); name = tok[0]
        emb = tok[-5] if len(tok) >= 7 else "?"
        if emb != "yes" or key not in name.replace(" ", ""): ok_fonts = False; bad.append(l)
    info = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
    w_mm = float(m.group(1)) * 25.4 / 72.0; h_mm = float(m.group(2)) * 25.4 / 72.0
    ok_w = abs(w_mm - WIDTH_MM) <= 0.5
    return dict(pdffonts=fonts, n_fonts=len(rows), fonts_ok=ok_fonts, bad_font_rows=bad, width_mm=w_mm, height_mm=h_mm, width_ok=ok_w)


def check_svg_fonts(svg_path):
    """Smallest font size written into the SVG (px at 72 dpi = pt) and the set of font families."""
    s = open(svg_path, encoding="utf-8").read()
    sizes = [float(x) for x in re.findall(r"font(?:-size)?:\s*(?:bold\s+|italic\s+|normal\s+)*([\d.]+)px", s)]
    fams = sorted(set(re.findall(r"px\s+'([^']+)'", s)) | set(re.findall(r"font-family:\s*'?([^;']+)'?", s)))
    return dict(min_font_pt=(min(sizes) if sizes else None), n_text=len(sizes), families=fams)


def finish(fig, n, font_name, meta: dict, data_rows: list, out=OUT):
    """Save all formats, run the PDF/SVG checks, write the data csv and a meta json for the report."""
    import pandas as pd
    paths = save_all(fig, n, out)
    df = pd.DataFrame(data_rows); df.to_csv(f"{out}/Dobbs_Figure{n}_data.csv", index=False); paths["csv"] = f"{out}/Dobbs_Figure{n}_data.csv"
    chk = check_pdf(paths["pdf"], font_name); svg = check_svg_fonts(paths["svg"])
    meta.update(font=FONT, pdf_check=chk, svg_check=svg, files={k: dict(path=p, sha256=sha256(p)) for k, p in paths.items()})
    assert chk["fonts_ok"], f"PDF font check failed:\n{chk['pdffonts']}"
    assert chk["width_ok"], f"PDF width {chk['width_mm']:.2f} mm != 170 ± 0.5"
    assert svg["min_font_pt"] is not None and svg["min_font_pt"] >= MIN_FS - 1e-6, f"SVG has text below {MIN_FS} pt: {svg}"
    json.dump(meta, open(f"{out}/fig{n}_meta.json", "w"), indent=1, default=str)
    print(f"Figure {n}: width {chk['width_mm']:.2f} mm, height {chk['height_mm']:.2f} mm, fonts ok={chk['fonts_ok']} ({chk['n_fonts']}), svg min font {svg['min_font_pt']} pt, families {svg['families']}")
    return paths
