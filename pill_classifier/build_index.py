"""
build_index.py
==============================================================
학습된 ArcFace 임베딩 모델로 알약 갤러리(전체 NDC) 임베딩을 만들고
FAISS 인덱스로 저장. front/back 둘 다 갤러리에 등록.
==============================================================
"""
import argparse
import csv
import pickle
from pathlib import Path

import faiss
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

IMG_SIZE = 224


class PillEmbedModel(nn.Module):
    def __init__(self, backbone_name, n_classes, emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
        self.embed = nn.Sequential(nn.Linear(feat_dim, emb_dim), nn.BatchNorm1d(emb_dim))

    def extract(self, x):
        return F.normalize(self.embed(self.backbone(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/arcface/best.pt")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="runs/arcface")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = PillEmbedModel(ckpt["backbone"], ckpt["n_train_classes"]).to(dev).eval()
    state = {k: v for k, v in ckpt["model"].items() if not k.startswith("head.")}
    model.load_state_dict(state, strict=False)

    tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    entries = []  # (ndc, view, path)
    with open(args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            entries.append((r["ndc"], "front", r["front_path"], r["split"]))
            entries.append((r["ndc"], "back", r["back_path"], r["split"]))

    embs = []
    with torch.inference_mode():
        for ndc, view, path, split in entries:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(dev)
            e = model.extract(x).cpu().numpy()[0]
            embs.append(e)
    embs = np.stack(embs).astype("float32")

    index = faiss.IndexFlatIP(embs.shape[1])  # 코사인 유사도 (정규화된 벡터의 내적)
    index.add(embs)
    faiss.write_index(index, str(Path(args.out) / "gallery.faiss"))
    with open(Path(args.out) / "gallery_meta.pkl", "wb") as f:
        pickle.dump(entries, f)

    print(f"gallery: {len(entries)} vectors -> {args.out}/gallery.faiss (+ gallery_meta.pkl)")


if __name__ == "__main__":
    main()
