"""
train_arcface.py
==============================================================
알약 외형(앞/뒤 이미지) ArcFace 임베딩 학습
  - 클래스당 실제 이미지 1장(앞)+1장(뒤)뿐 -> 강한 augmentation으로
    "같은 알약의 여러 뷰"를 epoch마다 다르게 생성해서 학습.
  - 학습 후 build_index.py로 FAISS 인덱스 생성, eval_retrieval.py로
    held-out 클래스(val) 검색 정확도 확인.
==============================================================
"""
import argparse
import csv
import math
import random
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMG_SIZE = 224


class ArcMarginProduct(nn.Module):
    """ArcFace head: cos(theta+m) margin on the true-class logit."""
    def __init__(self, in_dim, n_classes, s=30.0, m=0.30):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        x = F.normalize(x)
        w = F.normalize(self.weight)
        cosine = F.linear(x, w)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        onehot = torch.zeros_like(cosine)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = onehot * phi + (1.0 - onehot) * cosine
        return logits * self.s


class PillEmbedModel(nn.Module):
    def __init__(self, backbone_name, n_classes, emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
        self.embed = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.BatchNorm1d(emb_dim))
        self.head = ArcMarginProduct(emb_dim, n_classes)

    def extract(self, x):
        return F.normalize(self.embed(self.backbone(x)))

    def forward(self, x, labels):
        emb = self.embed(self.backbone(x))
        return self.head(emb, labels)


class PillDataset(Dataset):
    """epoch마다 클래스별로 front/back 중 하나를 무작위로 골라 강한 aug 적용."""
    def __init__(self, rows, train=True, views_per_class=20):
        self.rows = rows  # [(ndc_idx, front_path, back_path)]
        self.train = train
        self.views = views_per_class if train else 2  # val은 front+back 고정 2장

        train_aug = T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            T.RandomRotation(180),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        eval_tf = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.tf = train_aug if train else eval_tf

    def __len__(self):
        return len(self.rows) * self.views

    def __getitem__(self, i):
        cls_idx, front, back = self.rows[i % len(self.rows)]
        if self.train:
            path = front if random.random() < 0.5 else back
        else:
            path = front if (i // len(self.rows)) == 0 else back
        img = Image.open(path).convert("RGB")
        return self.tf(img), cls_idx


def load_manifest(path):
    rows_by_split = {"train": [], "val": []}
    ndc_to_idx = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ndc_to_idx.setdefault(r["ndc"], len(ndc_to_idx))
            rows_by_split[r["split"]].append((ndc_to_idx[r["ndc"]], r["front_path"], r["back_path"]))
    return rows_by_split, ndc_to_idx


def _embed_image(model, path, dev, eval_tf):
    img = Image.open(path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(dev)
    with torch.inference_mode():
        return model.extract(x)[0]


@torch.inference_mode()
def evaluate_retrieval(model, rows_by_split, dev):
    """front=query, back=gallery(전체) 로 top1/top5 검색 정확도 측정. (작은 데이터라 FAISS 없이 직접 계산)"""
    eval_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model.eval()
    all_rows = rows_by_split["train"] + rows_by_split["val"]
    gallery_cls, gallery_vecs = [], []
    front_by_cls = {}
    for cls_idx, front, back in all_rows:
        gallery_cls.append(cls_idx)
        gallery_vecs.append(_embed_image(model, back, dev, eval_tf))
        front_by_cls[cls_idx] = front
    gallery_vecs = torch.stack(gallery_vecs)  # (N, D), 이미 normalize됨

    results = {}
    for split in ("train", "val"):
        cls_list = sorted({r[0] for r in rows_by_split[split]})
        top1, top5, n = 0, 0, 0
        for cls_idx in cls_list:
            q = _embed_image(model, front_by_cls[cls_idx], dev, eval_tf)
            sims = gallery_vecs @ q  # cosine sim (정규화됐으므로 내적=코사인)
            ranked = torch.argsort(sims, descending=True)[:5]
            ranked_cls = [gallery_cls[i] for i in ranked.tolist()]
            n += 1
            top1 += (ranked_cls[0] == cls_idx)
            top5 += (cls_idx in ranked_cls)
        results[split] = (top1 / max(1, n), top5 / max(1, n))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--backbone", default="efficientnetv2_rw_s")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--views_per_class", type=int, default=20)
    ap.add_argument("--out", default="runs/arcface")
    ap.add_argument("--eval_every", type=int, default=5, help="N epoch마다 val 검색정확도 평가")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows_by_split, ndc_to_idx = load_manifest(args.manifest)
    n_classes = len(ndc_to_idx)
    print(f"classes(train+val 전체)={n_classes} train_rows={len(rows_by_split['train'])} val_rows={len(rows_by_split['val'])}")

    # ArcFace head는 train 클래스만 분류 대상으로 함 (val은 학습 중 본 적 없는 클래스로 유지)
    train_ndc_idx = sorted({r[0] for r in rows_by_split["train"]})
    remap = {old: new for new, old in enumerate(train_ndc_idx)}
    train_rows = [(remap[c], f, b) for c, f, b in rows_by_split["train"]]
    n_train_classes = len(train_ndc_idx)

    tr_ds = PillDataset(train_rows, train=True, views_per_class=args.views_per_class)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    model = PillEmbedModel(args.backbone, n_train_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ce = nn.CrossEntropyLoss()

    best_val_top1 = -1.0
    best_ep = -1
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x, y)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * x.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
            tot_n += x.size(0)
        sched.step()
        log_line = f"ep{ep:3d} loss={tot_loss/tot_n:.4f} train_acc={tot_correct/tot_n:.3f}"

        if ep % args.eval_every == 0 or ep == args.epochs:
            results = evaluate_retrieval(model, rows_by_split, dev)
            tr_top1, tr_top5 = results["train"]
            va_top1, va_top5 = results["val"]
            log_line += (f" | retrieval train_top1={tr_top1:.3f} train_top5={tr_top5:.3f}"
                         f" val_top1={va_top1:.3f} val_top5={va_top5:.3f}")
            ckpt = {"model": model.state_dict(), "backbone": args.backbone,
                    "ndc_to_idx": ndc_to_idx, "n_train_classes": n_train_classes,
                    "epoch": ep, "val_top1": va_top1, "val_top5": va_top5}
            torch.save(ckpt, Path(args.out) / f"ep{ep:03d}_valtop1={va_top1:.3f}.pt")
            if va_top1 > best_val_top1:
                best_val_top1 = va_top1
                best_ep = ep
                torch.save(ckpt, Path(args.out) / "best.pt")
                log_line += "  <- best 갱신"
        print(log_line)

    torch.save({"model": model.state_dict(), "backbone": args.backbone,
                "ndc_to_idx": ndc_to_idx, "n_train_classes": n_train_classes},
               Path(args.out) / "last.pt")
    print(f"saved -> {args.out}/last.pt")
    print(f"best val_top1={best_val_top1:.3f} @ epoch {best_ep} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
