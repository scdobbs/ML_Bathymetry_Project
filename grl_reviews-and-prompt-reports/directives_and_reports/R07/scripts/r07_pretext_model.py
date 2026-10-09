"""R07 §2.1-2.2 — the pretext network: the R06 HetUNetClean body with FiLM (r06_model.UNetBody, unchanged; module name
`body`, so its tensors carry the same keys as BlockHeadNet.body) and, for the P-V2 body, the R06 ship-product branch
(r06_model.ProductBranch, module name `branch`), plus two small heads that are discarded after pretraining:
  recon_head   1 x 1 conv on the decoder output at input resolution -> masked-cell reconstruction (§2.2 item 1)
  spread_head  the R06 block-head structure (central 128 px -> 4 x 4 pool -> 1 x 1 conv, SiLU, 1 x 1 conv, last layer
               zero-initialised; for P-V2 the pooled branch features are concatenated as in BlockHeadNet) -> standardised
               log10 ship_rsd per block (§2.2 item 2)
body_state_dict() returns exactly the body.* (+ branch.*) tensors, which train_r07.load_init loads into a fresh BlockHeadNet."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r07_common  # noqa: F401  (sys.path for r06_model; installs the guard)
import torch, torch.nn as nn, torch.nn.functional as F
from r06_model import UNetBody, ProductBranch, n_params

class PretextNet(nn.Module):
    def __init__(self, base_ch=32, ch_mult=(1, 2, 4, 4), in_ch=2, scale_dim=4, products=False, tile=128, margin=64, nb=4):
        super().__init__(); self.products, self.tile, self.margin, self.nb = products, tile, margin, nb
        self.body = UNetBody(base_ch, ch_mult, in_ch, scale_dim); c = self.body.out_ch
        if products: self.branch = ProductBranch(scale_dim=scale_dim); c += self.branch.out_ch
        self.recon_head = nn.Conv2d(self.body.out_ch, 1, 1)
        self.spread_head = nn.Sequential(nn.Conv2d(c, c, 1), nn.SiLU(), nn.Conv2d(c, 1, 1))
        nn.init.zeros_(self.spread_head[-1].weight); nn.init.zeros_(self.spread_head[-1].bias)
    def _central(self, h, full):          # BlockHeadNet._central (crop=False branch), verbatim
        f = h.shape[-1] / full; m, t = int(round(self.margin * f)), int(round(self.tile * f))
        return h[:, :, m:m + t, m:m + t]
    def forward(self, inp, cond, prod=None):
        full = inp.shape[-1]; h = self.body(inp, cond)
        recon = self.recon_head(h)[:, 0]
        f = F.adaptive_avg_pool2d(self._central(h, full), self.nb)
        if self.products:
            hp = self._central(self.branch(prod, cond), full); f = torch.cat([f, F.adaptive_avg_pool2d(hp, self.nb)], 1)
        return recon, self.spread_head(f)[:, 0]
    def body_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k.startswith(("body.", "branch."))}

if __name__ == "__main__":
    from r06_model import BlockHeadNet
    for prod in (False, True):
        net = PretextNet(products=prod); r, s = net(torch.zeros(2, 2, 256, 256), torch.zeros(2, 4), torch.zeros(2, 8, 256, 256) if prod else None)
        ref = BlockHeadNet(products=prod); sd = net.body_state_dict(); res = ref.load_state_dict(sd, strict=False)
        print(f"products={prod}: params {n_params(net):,} (body {n_params(net.body):,}; branch {n_params(net.branch) if prod else 0:,}; recon head {n_params(net.recon_head):,}; spread head {n_params(net.spread_head):,}) "
              f"out {tuple(r.shape)} {tuple(s.shape)}; -> BlockHeadNet: {len(sd)} tensors loaded, missing {res.missing_keys}, unexpected {res.unexpected_keys}")
