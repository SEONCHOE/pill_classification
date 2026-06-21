"""
drug_labels.json 생성 스크립트
로컬에서 1회 실행 후 생성된 JSON을 pill_multiagent/ 에 저장
"""

import pandas as pd
import json
import os

CSV_PATH = "C:/Users/SUN/Documents/Pill_classification/Drug_list/final_data/pill_ss_final.csv"

df = pd.read_csv(CSV_PATH, index_col=0)

labels = sorted(df["drug"].unique().tolist())

out_path = os.path.join(os.path.dirname(__file__), "drug_labels.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print(f"완료: {len(labels)}개 약물 레이블 → {out_path}")
print("샘플:", labels[:5])
