import re
from pathlib import Path

# RxImage filenames encode NDC like 00093-3109-05 (labeler-product-package)
# imprints often contain digits from the "product" (and sometimes labeler) segment.

lines = Path("ocr_check_full/_ocr_results.txt").read_text().splitlines()

total, hit, hit_highconf, total_highconf = 0, 0, 0, 0
for ln in lines:
    if not ln.strip():
        continue
    fname, confpart, ocrpart = ln.split("\t")
    conf = float(confpart.split("=")[1])
    ocr_text = ocrpart.split("=", 1)[1].strip("'")
    m = re.search(r"(\d{5})-(\d{4})-(\d{2})", fname)
    if not m:
        continue
    labeler, product, pkg = m.groups()
    ocr_digits = re.sub(r"\D", "", ocr_text)
    candidates = [product, product.lstrip("0"), labeler[-2:] + product, labeler[:2]]
    is_hit = any(c and len(c) >= 2 and c in ocr_digits for c in candidates)
    total += 1
    hit += is_hit
    if conf >= 0.3:
        total_highconf += 1
        hit_highconf += is_hit

print(f"전체 매칭률: {hit}/{total} = {hit/max(1,total):.3f}")
print(f"det_conf>=0.3 매칭률: {hit_highconf}/{total_highconf} = {hit_highconf/max(1,total_highconf):.3f}")
