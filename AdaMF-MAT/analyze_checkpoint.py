"""Post-hoc analysis of a trained AdvMixRotatE checkpoint (no training).

Run from the AdaMF-MAT root after building mmkgc (cd mmkgc && bash make.sh) and placing
./benchmarks/<DATASET>/ and ./embeddings/<DATASET>-{visual,textual}.pth.

  python analyze_checkpoint.py --ckpt checkpoint/mkgw_mu0 --mode attention
  python analyze_checkpoint.py --ckpt checkpoint/mkgw_mu0 --mode missing --out fig3_missing.csv

attention: mean modality weights (structural, visual, textual) over all entities (Figure 5).
missing:   zero the visual+textual features of a random fraction p of entities at test time
           and re-run link prediction for p in {0, .2, .4, .6, .8, 1} (Figure 3).
"""
import argparse
import csv

import torch

from mmkgc.config import Tester
from mmkgc.data import TestDataLoader, TrainDataLoader
from mmkgc.device import get_device
from mmkgc.module.model import AdvMixRotatE

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="MKG-W")
ap.add_argument("--dim", type=int, default=200)  # scripts/run_mkgw.sh
ap.add_argument("--ckpt", required=True)
ap.add_argument("--mode", choices=["attention", "missing"], required=True)
ap.add_argument("--out", default="fig3_missing.csv")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

bench = "./benchmarks/%s/" % args.dataset
train = TrainDataLoader(in_path=bench, batch_size=1024, threads=8, sampling_mode="normal",
                        bern_flag=1, filter_flag=1, neg_ent=128, neg_rel=0)
n_ent = train.get_ent_tot()
img = torch.load("./embeddings/%s-visual.pth" % args.dataset, map_location="cpu")
txt = torch.load("./embeddings/%s-textual.pth" % args.dataset, map_location="cpu")
model = AdvMixRotatE(ent_tot=n_ent, rel_tot=train.get_rel_tot(), dim=args.dim, margin=12.0,
                     epsilon=2.0, img_emb=img, text_emb=txt, ablation=None)
model.load_checkpoint(args.ckpt)
dev = get_device()
model.to(dev).eval()

if args.mode == "attention":
    with torch.no_grad():
        idx = torch.arange(n_ent, device=dev)
        es = model.ent_embeddings(idx)
        ev = model.img_proj(model.img_embeddings(idx))
        et = model.text_proj(model.text_embeddings(idx))
        s, v, t = model.get_attention(es, ev, et).mean(0).tolist()
    print("structural=%.4f visual=%.4f textual=%.4f" % (s, v, t))
else:
    test = TestDataLoader(bench, "link")
    orig_img = model.img_embeddings.weight.data.clone()
    orig_txt = model.text_embeddings.weight.data.clone()
    g = torch.Generator().manual_seed(args.seed)
    rows = []
    for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        model.img_embeddings.weight.data.copy_(orig_img)
        model.text_embeddings.weight.data.copy_(orig_txt)
        if p > 0:
            mask = (torch.rand(n_ent, generator=g) < p).to(dev)
            model.img_embeddings.weight.data[mask] = 0.0
            model.text_embeddings.weight.data[mask] = 0.0
        mrr, mr, h10, h3, h1 = Tester(model=model, data_loader=test,
                                      use_gpu=True).run_link_prediction(type_constrain=False)
        print("missing=%.1f MRR=%.4f H@10=%.4f H@3=%.4f H@1=%.4f" % (p, mrr, h10, h3, h1))
        rows.append([p, float(mrr), float(mr), float(h10), float(h3), float(h1)])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["missing_rate", "MRR", "MR", "hit@10", "hit@3", "hit@1"])
        w.writerows(rows)
    print("saved", args.out)
