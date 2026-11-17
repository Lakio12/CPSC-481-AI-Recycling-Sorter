import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

from dataset import get_dataloaders


ROOT = Path(__file__).resolve().parent.parent
MPTH = ROOT / "Models" / "efficientnet_b0_trashnet.pth"
HPTH = ROOT / "Reports" / "train_history.csv"

SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EPS = 15
BS = 32
LR = 3e-4
WD = 1e-4
LS = 0.1
SEED = 42


def bld(nc=6, pre=True):
    wts = EfficientNet_B0_Weights.IMAGENET1K_V1 if pre else None
    mdl = efficientnet_b0(weights=wts)
    inf = mdl.classifier[-1].in_features
    mdl.classifier[-1] = nn.Linear(inf, nc)
    return mdl


def run(mdl, ldr, crit, dev, opt=None, dsc=None):
    trn = opt is not None
    mdl.train(trn)

    tot = 0.0
    cor = 0
    seen = 0

    bat = tqdm(ldr, desc=dsc, leave=False) if dsc else ldr
    with torch.set_grad_enabled(trn):
        for img, lbl in bat:
            img, lbl = img.to(dev), lbl.to(dev)
            outs = mdl(img)
            loss = crit(outs, lbl)

            if trn:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            tot += loss.item() * lbl.size(0)
            cor += (outs.argmax(dim=1) == lbl).sum().item()
            seen += lbl.size(0)

    return tot / seen, cor / seen


def save(mdl, pth, cls, **meta):
    pth = Path(pth)
    pth.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "arch": "efficientnet_b0",
        "img_size": SIZE,
        "mean": MEAN,
        "std": STD,
        "classes": list(cls),
        "model_state": mdl.state_dict(),
        **meta,
    }
    torch.save(ckpt, pth)
    return pth


def fit(trl, vl, cls, eps=EPS, lr=LR, dev=None, out=MPTH, pre=True, wd=WD, ls=LS, hp=None):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")

    mdl = bld(nc=len(cls), pre=pre).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=ls)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=eps)

    hist = []
    best = -1.0

    for ep in range(1, eps + 1):
        t0 = time.time()
        tl, ta = run(mdl, trl, crit, dev, opt, dsc=f"epoch {ep}/{eps} train")
        vls, va = run(mdl, vl, crit, dev, dsc=f"epoch {ep}/{eps} val")
        clr = opt.param_groups[0]["lr"]
        sch.step()

        rec = {
            "epoch": ep,
            "train_loss": tl,
            "train_acc": ta,
            "val_loss": vls,
            "val_acc": va,
            "lr": clr,
            "seconds": time.time() - t0,
        }
        hist.append(rec)

        imp = va > best
        if imp:
            best = va
            save(mdl, out, cls, **{"val_acc": va, "epoch": ep})

        mark = "  * saved" if imp else ""
        print(
            f"epoch {ep:>2}/{eps}  "
            f"train loss {tl:.4f} acc {ta:.4f}  |  "
            f"val loss {vls:.4f} acc {va:.4f}  "
            f"({rec['seconds']:.0f}s){mark}"
        )

    if hp is not None:
        hp = Path(hp)
        hp.parent.mkdir(parents=True, exist_ok=True)
        with open(hp, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            wr.writeheader()
            wr.writerows(hist)

    return hist


def main():
    par = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 on TrashNet")
    par.add_argument("--epochs", dest="eps", type=int, default=EPS)
    par.add_argument("--batch-size", dest="bs", type=int, default=BS)
    par.add_argument("--lr", dest="lr", type=float, default=LR)
    par.add_argument("--seed", dest="seed", type=int, default=SEED)
    par.add_argument("--out", dest="out", type=Path, default=MPTH)
    args = par.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""
    print(f"Device: {dev}{gpu}")

    trl, vl, _, cls = get_dataloaders(batch_size=args.bs)
    print(f"Classes: {cls}")
    print(f"Training images: {len(trl.dataset)}, validation images: {len(vl.dataset)}\n")

    hist = fit(trl, vl, cls, eps=args.eps, lr=args.lr, dev=dev, out=args.out, hp=HPTH)

    top = max(hist, key=lambda h: h["val_acc"])
    print(f"\nBest validation accuracy {top['val_acc']:.4f} at epoch {top['epoch']}")
    print(f"Checkpoint: {args.out}")
    print(f"History:    {HPTH}")


if __name__ == "__main__":
    main()
