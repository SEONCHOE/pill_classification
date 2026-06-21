from ultralytics import YOLO

BEST = "/home/user/Github/runs/detect/runs/imprint/yolo26n_p2_detect-3/weights/best.pt"
DATA = "imprint_det.yaml"

model = YOLO(BEST)

print("\n===== (1) Per-class val (default conf, best-F1 threshold) =====")
metrics = model.val(data=DATA, imgsz=960, split="val", verbose=True)
names = metrics.names
print("class_index ->", metrics.box.ap_class_index)
for i, ci in enumerate(metrics.box.ap_class_index):
    print(f"class={names[ci]:12s} P={metrics.box.p[i]:.3f} R={metrics.box.r[i]:.3f} AP50={metrics.box.ap50[i]:.3f} AP={metrics.box.ap[i]:.3f}")
print(f"overall: P={metrics.box.mp:.3f} R={metrics.box.mr:.3f} mAP50={metrics.box.map50:.3f} mAP={metrics.box.map:.3f}")

print("\n===== (2) Max-recall check: conf=0.001, loose IoU =====")
metrics_lowconf = model.val(data=DATA, imgsz=960, split="val", conf=0.001, iou=0.3, verbose=False)
for i, ci in enumerate(metrics_lowconf.box.ap_class_index):
    print(f"class={names[ci]:12s} P={metrics_lowconf.box.p[i]:.3f} R={metrics_lowconf.box.r[i]:.3f}")
print(f"overall (conf=0.001, iou=0.3): P={metrics_lowconf.box.mp:.3f} R={metrics_lowconf.box.mr:.3f} mAP50={metrics_lowconf.box.map50:.3f}")
