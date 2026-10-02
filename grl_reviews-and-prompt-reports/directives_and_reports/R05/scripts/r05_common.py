"""R05 shared definitions: reuses r04_common (guard, manifest, frozen unit list) and adds the R05 roots."""
import os, sys
R04_SCRIPTS = "/home/users/scdobbs/GAN_Bathymetry_Project/grl_reviews-and-prompt-reports/directives_and_reports/R04/scripts"
R05_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (R04_SCRIPTS, R05_SCRIPTS):
    if p not in sys.path: sys.path.insert(0, p)
import r04_common as C4
ROOT = "/scratch/users/scdobbs/grl_review/R05"; R04 = C4.ROOT
TILE_FLOOR = 0.5          # §2.1: tile kept iff >= 50 % of its 128 x 128 px are joint-valid
MIN_TILES_PER_PAIR = 10   # §2.2.4 pair eligibility
DETECTION_BOUNDS = (0.05, 0.95)   # §2.2.5, per pair, block-scale prevalence of E_1m > 0
def start(name): return C4.start(name)
