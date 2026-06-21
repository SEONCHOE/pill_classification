"""
train_str.py
==============================================================
각인 STR 학습 파이프라인 (CRNN + CTC)
  - 입력: auto_label_gemini.py가 만든 str_manifest.csv (crop_path,text)
  - 자체 완결형(PyTorch만 있으면 실행). 모바일 친화 경량 baseline.
  - 약물사전 lexicon 보정 후처리 포함.
  - 더 높은 정확도가 필요하면 PARSeq로 교체(맨 아래 주석 참고).

설치:
    pip install -U torch torchvision pillow numpy

실행:
    python train_str.py --manifest ./datasets/imprint/str_manifest.csv \
                        --lexicon ./drug_imprints.txt \
                        --epochs 100 --out ./runs/str

데이터:
    str_manifest.csv : crop_path,text
    drug_imprints.txt: 약물사전의 각인 문자열, 한 줄에 하나 (lexicon 보정용)
==============================================================
"""
import os, csv, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMG_H, IMG_W = 32, 128            # 각인 crop 정규화 크기

# ----------------------------------------------------------------
# 1. 문자 집합(charset) — 영숫자 기본. 한글 각인 있으면 확장.
# ----------------------------------------------------------------
def build_charset(texts, extra=""):
    chars = set(extra)
    for t in texts:
        chars.update(list(t))
    charset = sorted(chars)
    # index 0 = CTC blank. 1..N = 실제 문자
    stoi = {c: i + 1 for i, c in enumerate(charset)}
    itos = {i + 1: c for i, c in enumerate(charset)}
    return charset, stoi, itos

# ----------------------------------------------------------------
# 2. 데이터셋
# ----------------------------------------------------------------
class STRDataset(Dataset):
    def __init__(self, rows, stoi, train=True):
        self.rows, self.stoi = rows, stoi
        aug = [T.RandomRotation(8, fill=255),
               T.ColorJitter(brightness=0.4, contrast=0.4)] if train else []
        self.tf = T.Compose([T.Grayscale(),
                             T.Resize((IMG_H, IMG_W)),
                             *aug,
                             T.ToTensor(),
                             T.Normalize([0.5], [0.5])])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        path, text = self.rows[i]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor([self.stoi[c] for c in text if c in self.stoi], dtype=torch.long)
        return x, y, text

def collate(batch):
    xs, ys, texts = zip(*batch)
    xs = torch.stack(xs)
    target_lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    targets = torch.cat(ys) if sum(len(y) for y in ys) > 0 else torch.zeros(0, dtype=torch.long)
    return xs, targets, target_lengths, list(texts)

# ----------------------------------------------------------------
# 3. CRNN 모델 (CNN -> 높이 평균 -> BiLSTM -> CTC)
# ----------------------------------------------------------------
class CRNN(nn.Module):
    def __init__(self, n_class, n_hidden=256):
        super().__init__()
        def block(i, o, k=3, s=1, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, s, p), nn.BatchNorm2d(o), nn.ReLU(True))
        self.cnn = nn.Sequential(
            block(1, 64),   nn.MaxPool2d(2, 2),              # H/2  W/2
            block(64, 128), nn.MaxPool2d(2, 2),              # H/4  W/4
            block(128, 256), block(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),                    # 높이만 절반
            block(256, 512), block(512, 512),
            nn.MaxPool2d((2, 1), (2, 1)),                    # 높이만 절반
        )
        self.rnn = nn.LSTM(512, n_hidden, num_layers=2,
                           bidirectional=True, batch_first=False)
        self.fc = nn.Linear(n_hidden * 2, n_class)           # n_class = charset+1(blank)

    def forward(self, x):
        f = self.cnn(x)                  # (B, C, H', W')
        f = f.mean(dim=2)                # 높이 평균 -> (B, C, W')  (height 산술 견고)
        f = f.permute(2, 0, 1)           # (T=W', B, C)
        f, _ = self.rnn(f)
        return self.fc(f)                # (T, B, n_class)

# ----------------------------------------------------------------
# 4. 디코딩 / 지표
# ----------------------------------------------------------------
def greedy_decode(logits, itos):
    """logits: (T,B,n_class) -> list[str]"""
    idx = logits.argmax(2).permute(1, 0).cpu().numpy()   # (B,T)
    out = []
    for seq in idx:
        prev, s = 0, []
        for p in seq:
            if p != 0 and p != prev:
                s.append(itos.get(int(p), ""))
            prev = p
        out.append("".join(s))
    return out

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (a[i-1] != b[j-1]))
            prev = cur
    return dp[n]

def cer(pred, gt):
    return levenshtein(pred, gt) / max(1, len(gt))

# ----------------------------------------------------------------
# 5. Lexicon 보정 (약물사전 closed-set)
# ----------------------------------------------------------------
def lexicon_correct(pred, lexicon, max_dist=2):
    """예측을 약물사전 각인 후보 중 최근접으로 교정(거리 임계 내)."""
    if not lexicon or pred in lexicon:
        return pred
    best, bd = pred, max_dist + 1
    for cand in lexicon:
        d = levenshtein(pred, cand)
        if d < bd:
            best, bd = cand, d
    return best if bd <= max_dist else pred

# ----------------------------------------------------------------
# 6. 학습 루프
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lexicon", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="./runs/str")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--extra_chars", default="", help="추가 문자(예: 한글). 자동 수집 외 강제 포함")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터 로드
    rows = []
    with open(args.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["text"].strip():
                rows.append((r["crop_path"], r["text"].strip()))
    random.seed(42); random.shuffle(rows)
    n_val = int(len(rows) * args.val_ratio)
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    lexicon = []
    if args.lexicon and os.path.exists(args.lexicon):
        lexicon = [l.strip() for l in open(args.lexicon, encoding="utf-8") if l.strip()]

    charset, stoi, itos = build_charset([t for _, t in rows], extra=args.extra_chars)
    n_class = len(charset) + 1
    print(f"train={len(train_rows)} val={len(val_rows)} charset={len(charset)} lexicon={len(lexicon)}")

    tr = DataLoader(STRDataset(train_rows, stoi, True), batch_size=args.batch,
                    shuffle=True, collate_fn=collate, num_workers=4)
    va = DataLoader(STRDataset(val_rows, stoi, False), batch_size=args.batch,
                    shuffle=False, collate_fn=collate, num_workers=4)

    model = CRNN(n_class).to(dev)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, targets, tlens, _ in tr:
            x, targets, tlens = x.to(dev), targets.to(dev), tlens.to(dev)
            logits = model(x)                                  # (T,B,n_class)
            logp = logits.log_softmax(2)
            ilens = torch.full((x.size(0),), logits.size(0), dtype=torch.long, device=dev)
            loss = ctc(logp, targets, ilens, tlens)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        # 검증
        model.eval()
        exact, exact_lex, cer_sum, n = 0, 0, 0.0, 0
        with torch.no_grad():
            for x, _, _, texts in va:
                preds = greedy_decode(model(x.to(dev)), itos)
                for pr, gt in zip(preds, texts):
                    exact += (pr == gt)
                    exact_lex += (lexicon_correct(pr, lexicon) == gt)
                    cer_sum += cer(pr, gt); n += 1
        acc, acc_lex, mcer = exact/max(1,n), exact_lex/max(1,n), cer_sum/max(1,n)
        print(f"ep{ep:3d} acc={acc:.3f} acc+lex={acc_lex:.3f} CER={mcer:.3f}")

        if acc_lex > best_acc:
            best_acc = acc_lex
            torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos,
                        "charset": charset}, Path(args.out) / "best.pt")

    print(f"best acc+lex = {best_acc:.3f}  -> {args.out}/best.pt")
    # 모바일 export 예시:
    #   dummy = torch.randn(1,1,IMG_H,IMG_W)
    #   torch.onnx.export(model.cpu(), dummy, f"{args.out}/str.onnx", opset_version=17)

# ----------------------------------------------------------------
# [더 높은 정확도] PARSeq로 교체하려면:
#   1) git clone https://github.com/baudm/parseq
#   2) crops/ + text를 PARSeq의 LMDB 포맷으로 변환 (repo의 tools 사용)
#   3) ./train.py +experiment=parseq dataset=real  로 fine-tune
#      (charset에 한글 추가 시 charset 설정 수정)
#   4) 추론 출력에 위 lexicon_correct()를 동일하게 후처리로 적용
# 곡면/음각 비중이 높으면 SVTRv2-T도 동일 자리에서 교체 가능.
# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
