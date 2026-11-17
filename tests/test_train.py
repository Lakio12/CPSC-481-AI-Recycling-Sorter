import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train import bld, fit, run, save


def fake(n=8, nc=6, size=32, bs=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(n, 3, size, size, generator=g)
    lbl = torch.arange(n) % nc
    return DataLoader(TensorDataset(img, lbl), batch_size=bs)


class Zero(nn.Module):
    def __init__(self, nc=6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(nc))

    def forward(self, x):
        outs = torch.full((x.shape[0], self.bias.numel()), -5.0)
        outs[:, 0] = 5.0
        return outs + self.bias


@pytest.mark.parametrize("nc", [6, 3])
def test_bld_outputs_one_score_per_class(nc):
    mdl = bld(nc=nc, pre=False)
    mdl.eval()
    outs = mdl(torch.zeros(2, 3, 224, 224))
    assert outs.shape == (2, nc)


def test_bld_is_efficientnet_b0():
    mdl = bld(nc=6, pre=False)
    head = mdl.classifier[-1]
    assert isinstance(head, nn.Linear)
    assert head.in_features == 1280
    assert head.out_features == 6


def test_run_computes_accuracy_from_predictions():
    mdl = Zero()
    loss, acc = run(mdl, fake(n=8), nn.CrossEntropyLoss(), dev="cpu")
    assert acc == pytest.approx(2 / 8)
    assert loss > 0


def test_run_with_optimizer_updates_weights():
    torch.manual_seed(0)
    mdl = bld(nc=6, pre=False)
    old = mdl.classifier[-1].weight.detach().clone()
    opt = torch.optim.SGD(mdl.parameters(), lr=0.1)
    run(mdl, fake(), nn.CrossEntropyLoss(), dev="cpu", opt=opt)
    assert not torch.equal(old, mdl.classifier[-1].weight)


def test_run_without_optimizer_leaves_weights_unchanged():
    torch.manual_seed(0)
    mdl = bld(nc=6, pre=False)
    old = [p.detach().clone() for p in mdl.parameters()]
    run(mdl, fake(), nn.CrossEntropyLoss(), dev="cpu")
    new = list(mdl.parameters())
    assert all(torch.equal(a, b) for a, b in zip(old, new))


def test_save_roundtrip_restores_identical_predictions(tmp_path):
    torch.manual_seed(0)
    cls = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    mdl = bld(nc=6, pre=False).eval()
    x = torch.rand(1, 3, 224, 224)
    exp = mdl(x)

    pth = save(mdl, tmp_path / "Models" / "m.pth", cls, val_acc=0.5)

    assert pth.exists()
    ckpt = torch.load(pth, map_location="cpu")
    assert ckpt["classes"] == cls
    assert ckpt["arch"] == "efficientnet_b0"
    assert ckpt["img_size"] == 224
    assert ckpt["val_acc"] == 0.5

    new = bld(nc=6, pre=False)
    new.load_state_dict(ckpt["model_state"])
    new.eval()
    assert torch.allclose(new(x), exp)


def test_fit_saves_best_checkpoint_and_history(tmp_path):
    torch.manual_seed(0)
    cls = ["a", "b", "c", "d", "e", "f"]
    out = tmp_path / "best.pth"

    hist = fit(fake(n=8, seed=1), fake(n=8, seed=2), cls, eps=2, lr=1e-3, dev="cpu", out=out, pre=False)

    assert len(hist) == 2
    assert {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"} <= set(hist[0])
    assert out.exists()
    ckpt = torch.load(out, map_location="cpu")
    assert ckpt["classes"] == cls
    assert ckpt["val_acc"] == max(h["val_acc"] for h in hist)
