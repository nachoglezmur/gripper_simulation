"""CNN de punto de agarre por behavior cloning (imagen -> XYZ).

ResNet18 pretrained + cabeza Linear(512->3). Demos de expert_demos.py.
Criterio: error medio XY en test < 2 cm. Exporta ONNX.

Uso:
  python -m control.policy_bc --epochs 50            # entreno completo
  python -m control.policy_bc --overfit              # sanity: 8 muestras
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models


class DemoSet(Dataset):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, images, picks_xy):
        self.images = images
        self.picks = picks_xy.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        return (img - self.MEAN) / self.STD, torch.from_numpy(self.picks[i])


class GraspCNN(nn.Module):
    """Imagen -> (x, y) de agarre en metros. z siempre es Z_PICK constante."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.backbone(x)


def train_epoch(net, loader, opt, loss_fn, dev):
    net.train()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
        tot += loss.item() * len(x)
        n += len(x)
    return tot / n


@torch.no_grad()
def eval_xy(net, loader, dev, mean, std):
    net.eval()
    errs = []
    mean_t = torch.from_numpy(mean).to(dev)
    std_t = torch.from_numpy(std).to(dev)
    for x, y in loader:
        pred = net(x.to(dev)).cpu() * std_t.cpu() + mean_t.cpu()
        errs.append((pred - y).norm(dim=1))
    return torch.cat(errs).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--demos", type=str, default="data/demos/demos.npz",
                    help="uno o varios .npz separados por coma")
    ap.add_argument("--out", type=str, default="models")
    ap.add_argument("--export-only", action="store_true",
                    help="solo exporta ONNX desde el .pt existente")
    args = ap.parse_args()

    import pathlib

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    assert torch.cuda.is_available()

    if args.export_only:
        net = GraspCNN().to(dev)
        net.load_state_dict(torch.load(f"{args.out}/grasp_cnn.pt",
                                       map_location=dev, weights_only=True))
        net.eval()
        torch.onnx.export(
            net, torch.randn(1, 3, 480, 640, device=dev),
            f"{args.out}/grasp_cnn.onnx", input_names=["image"],
            output_names=["pick_xy"], opset_version=17,
            dynamic_axes={"image": {0: "batch"}, "pick_xy": {0: "batch"}},
        )
        print(f"OK: {args.out}/grasp_cnn.onnx", flush=True)
        return

    imgs, picks = [], []
    for p in args.demos.split(","):
        d = np.load(p.strip())
        imgs.append(d["images"])
        picks.append(d["pick_xyz"][:, :2])  # solo XY; z = Z_PICK constante
    images = np.concatenate(imgs)
    picks_xy = np.concatenate(picks)
    print(f"demos: {len(images)}", flush=True)
    ds = DemoSet(images, picks_xy)
    if args.overfit:
        ds = torch.utils.data.Subset(ds, list(range(8)))
        train_ds, test_ds = ds, ds
        epochs = 30
    else:
        n_test = max(1, int(0.1 * len(ds)))
        train_ds, test_ds = random_split(
            ds, [len(ds) - n_test, n_test],
            generator=torch.Generator().manual_seed(7),
        )
        epochs = args.epochs
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=2,
                             pin_memory=True)

    net = GraspCNN().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    # Estandarización de targets con stats del TRAIN (se guardan para inferencia).
    train_idx = train_ds.indices if hasattr(train_ds, "indices") else list(range(len(train_ds)))
    tmean = picks_xy[np.array(train_idx)].mean(axis=0)
    tstd = picks_xy[np.array(train_idx)].std(axis=0) + 1e-6
    np.savez(f"{args.out}/grasp_norm.npz", mean=tmean, std=tstd)
    mean_t = torch.from_numpy(tmean).float()
    std_t = torch.from_numpy(tstd).float()

    def standardize(batch_y):
        return (batch_y - mean_t) / std_t

    for ep in range(epochs):
        net.train()
        tot, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(dev)
            ys = standardize(y).to(dev)
            opt.zero_grad()
            loss = loss_fn(net(x), ys)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        sched.step()
        loss = tot / n
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep {ep + 1}/{epochs} loss={loss:.6f}", flush=True)

    if args.overfit:
        assert loss < 1e-3, f"no sobreajusta: loss={loss:.6f}"
        print("OVERFIT OK", flush=True)
        return

    err = eval_xy(net, test_loader, dev, tmean, tstd)
    print(f"error medio XY test = {err * 100:.2f} cm", flush=True)
    assert err < 0.02, f"error {err:.4f} m >= 2 cm"
    torch.save(net.state_dict(), f"{args.out}/grasp_cnn.pt")
    net.eval()
    torch.onnx.export(
        net, torch.randn(1, 3, 480, 640, device=dev),
        f"{args.out}/grasp_cnn.onnx", input_names=["image"],
        output_names=["pick_xy"], opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "pick_xy": {0: "batch"}},
    )
    print(f"OK: {args.out}/grasp_cnn.pt + .onnx + grasp_norm.npz", flush=True)


class GraspPolicy:
    """Inferencia del punto de agarre: imagen RGB 480x640 -> (x, y) metros."""

    def __init__(self, weights="models/grasp_cnn.pt",
                 norm="models/grasp_norm.npz", device="cuda"):
        self.dev = torch.device(device)
        self.net = GraspCNN().to(self.dev)
        self.net.load_state_dict(
            torch.load(weights, map_location=self.dev, weights_only=True))
        self.net.eval()
        stats = np.load(norm)
        self.mean = torch.from_numpy(stats["mean"]).float().to(self.dev)
        self.std = torch.from_numpy(stats["std"]).float().to(self.dev)

    @torch.no_grad()
    def predict(self, image_rgb):
        import numpy as np

        x = (torch.from_numpy(np.asarray(image_rgb)).permute(2, 0, 1).float()
             / 255.0)
        x = (x - DemoSet.MEAN) / DemoSet.STD
        out = self.net(x.unsqueeze(0).to(self.dev)).squeeze(0)
        return (out * self.std + self.mean).cpu().numpy()


if __name__ == "__main__":
    main()
