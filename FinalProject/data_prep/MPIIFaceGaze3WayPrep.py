"""
ZIP FILE SOURCE: https://collaborative-ai.org/files/datasets/MPIIFaceGaze.zip


MPIIFaceGaze 3-class (left/straight/right) pipeline - standalone, from scratch
=================================================================================
 
Goes directly from the raw unzipped MPIIFaceGaze dump (participants p00-p14,
each with dayNN/ image folders + a pXX.txt annotation file) to 224x224 face
crops labeled left/straight/right, with no dependency on any other script or
pre-built manifest. Only the horizontal gaze signal (yaw) is used - vertical
gaze (pitch) is ignored entirely, so a frame where someone was looking
down-and-right still gets labeled "right", not dropped or reclassified.
 
Annotation format (pXX.txt, one row per image, whitespace separated, 28
columns - see MPIIFaceGaze's own readme.txt for the authoritative spec):
    0        image path, relative to the participant folder (e.g. day01/0005.jpg)
    1-2      on-screen gaze point, in pixels (unused here)
    3-14     6 facial landmarks (x,y) - 4 eye corners + 2 mouth corners
    15-20    estimated 3D head pose (unused here)
    21-23    fc: 3D face center in the camera coordinate system
    24-26    gt: 3D gaze target location in the camera coordinate system
    27       which eye was used in the original paper's eval subset (unused here)
 
Cropping:
MPIIFaceGaze doesn't ship a precomputed face box. It gives 6 facial
landmarks (eye + mouth corners), so we build a box from their extent, then
expand it asymmetrically (more above the eyes for forehead, less below the
mouth for chin, since the landmarks only span eye-to-mouth) to approximate a
full-face crop.
 
Labeling:
gt - fc is the 3D gaze direction vector, in the camera's coordinate system.
We normalize it and take yaw_deg = degrees(atan2(-x, -z)), with +yaw = right
- verified empirically by correlating this formula against the dataset's own
on-screen gaze pixel coordinates (screen_x) before trusting the sign
convention (r ~ 0.97-0.98 across multiple subjects).
 
    |yaw_deg| <= --threshold degrees  -> straight
    yaw_deg >  threshold              -> right
    yaw_deg < -threshold              -> left
 
Unlike GazeCapture's XCam (a linear cm distance that implicitly assumes a
fixed, uncalibrated viewing distance), yaw_deg is a true visual angle derived
from 3D geometry, so it isn't subject to that same distance-dependent bias.
 
Split: subject-level 80/20 - each subject assigned entirely to train or test
(never split across both), so a person's face never leaks across the split.
MPIIFaceGaze ships no official split of its own (it's normally used with
leave-one-subject-out cross-validation), so this 80/20 split is invented by
this script, not something inherited from the dataset.
 
Output layout:
    <output_dir>/train/<label>/<subject>_<day>_<frame>.jpg
    <output_dir>/test/<label>/<subject>_<day>_<frame>.jpg
    <output_dir>/manifest.csv
 
Usage:
    python3 mpiifacegaze_3class.py --input_dir /path/to/MPIIFaceGaze --output_dir /path/to/out
    python3 mpiifacegaze_3class.py --input_dir ... --output_dir ... --limit_subjects 2  # dry run
"""
 
import argparse
import csv
import math
import os
import random
import sys
from multiprocessing import Pool, cpu_count
 
from PIL import Image
 
LABELS = ["left", "straight", "right"]
 
 
def classify_horizontal(yaw_deg, threshold):
    if abs(yaw_deg) <= threshold:
        return "straight"
    return "right" if yaw_deg > 0 else "left"
 
 
def find_subject_dirs(input_dir):
    subjects = []
    for entry in sorted(os.listdir(input_dir)):
        sdir = os.path.join(input_dir, entry)
        ann = os.path.join(sdir, f"{entry}.txt")
        if os.path.isdir(sdir) and os.path.exists(ann):
            subjects.append((entry, sdir, ann))
    return subjects
 
 
def parse_annotation_line(line):
    parts = line.split()
    if len(parts) != 28:
        return None
    rel_path = parts[0]
    lm_x = [float(parts[i]) for i in (3, 5, 7, 9, 11, 13)]
    lm_y = [float(parts[i]) for i in (4, 6, 8, 10, 12, 14)]
    fc = [float(v) for v in parts[21:24]]
    gt = [float(v) for v in parts[24:27]]
    return {"rel_path": rel_path, "lm_x": lm_x, "lm_y": lm_y, "fc": fc, "gt": gt}
 
 
def gaze_to_yaw(fc, gt):
    g = [gt[i] - fc[i] for i in range(3)]
    norm = math.sqrt(sum(v * v for v in g))
    if norm == 0:
        return 0.0
    gx, gy, gz = (v / norm for v in g)
    return math.degrees(math.atan2(-gx, -gz))
 
 
def process_subject(args):
    (subject_id, subject_dir, ann_path, output_dir, img_size,
     side_margin, top_margin, bottom_margin, threshold, split) = args
    rows = []
    errors = []
    try:
        with open(ann_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return [], [f"{subject_id}: failed to read annotation file ({e})"]
 
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = parse_annotation_line(line)
        if rec is None:
            errors.append(f"{subject_id}: malformed annotation line, skipped")
            continue
        try:
            src_path = os.path.join(subject_dir, rec["rel_path"])
            if not os.path.exists(src_path):
                errors.append(f"{subject_id}/{rec['rel_path']}: image file missing")
                continue
 
            yaw_deg = gaze_to_yaw(rec["fc"], rec["gt"])
            label = classify_horizontal(yaw_deg, threshold)
 
            min_x, max_x = min(rec["lm_x"]), max(rec["lm_x"])
            min_y, max_y = min(rec["lm_y"]), max(rec["lm_y"])
            w, h = max_x - min_x, max_y - min_y
 
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                img_w, img_h = im.size
                left = max(0, min_x - side_margin * w)
                right = min(img_w, max_x + side_margin * w)
                top = max(0, min_y - top_margin * h)
                bottom = min(img_h, max_y + bottom_margin * h)
                if right <= left or bottom <= top:
                    errors.append(f"{subject_id}/{rec['rel_path']}: degenerate crop box, skipped")
                    continue
                crop = im.crop((left, top, right, bottom)).resize((img_size, img_size), Image.LANCZOS)
 
                out_dir = os.path.join(output_dir, split, label)
                os.makedirs(out_dir, exist_ok=True)
                day, base = rec["rel_path"].split("/", 1)
                base_name = os.path.splitext(os.path.basename(base))[0]
                out_name = f"{subject_id}_{day}_{base_name}.jpg"
                out_path = os.path.join(out_dir, out_name)
                crop.save(out_path, "JPEG", quality=95)
 
            rows.append({
                "subject": subject_id,
                "day": day,
                "frame": os.path.basename(rec["rel_path"]),
                "split": split,
                "label": label,
                "yaw_deg": yaw_deg,
                "out_path": os.path.relpath(out_path, output_dir),
            })
        except Exception as e:
            errors.append(f"{subject_id}/{rec.get('rel_path','?')}: {e}")
 
    return rows, errors
 
 
def split_subjects(subjects, train_frac, seed):
    subjects = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    cut = int(round(len(subjects) * train_frac))
    return set(subjects[:cut]), set(subjects[cut:])
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="Path to unzipped MPIIFaceGaze folder (contains p00..p14)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--side_margin", type=float, default=0.55)
    ap.add_argument("--top_margin", type=float, default=1.2)
    ap.add_argument("--bottom_margin", type=float, default=0.75)
    ap.add_argument("--threshold", type=float, default=5.0, help="degrees, default 5.0")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--limit_subjects", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--skip_existing", action="store_true", help="Skip a subject if already in manifest.csv (safe to resume/chunk)")
    args = ap.parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
    subjects = find_subject_dirs(args.input_dir)
    all_ids = [s[0] for s in subjects]
    train_subjects, test_subjects = split_subjects(all_ids, args.train_frac, args.seed)
 
    subjects = subjects[args.offset:]
    if args.limit_subjects:
        subjects = subjects[: args.limit_subjects]
    if not subjects:
        print(f"No subject folders found (after offset {args.offset})", file=sys.stderr)
        sys.exit(1)
 
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = ["subject", "day", "frame", "split", "label", "yaw_deg", "out_path"]
    manifest_exists = os.path.exists(manifest_path)
    already_done = set()
    if args.skip_existing and manifest_exists:
        with open(manifest_path, "r", newline="") as mf:
            for row in csv.DictReader(mf):
                already_done.add(row["subject"])
        subjects = [(sid, sdir, ann) for sid, sdir, ann in subjects if sid not in already_done]
 
    print(f"Subjects total: {len(all_ids)} ({len(train_subjects)} train / {len(test_subjects)} test). "
          f"Processing {len(subjects)} this run (already_done_skipped={len(already_done)}), {args.workers} workers, threshold={args.threshold}deg")
 
    if not subjects:
        print("Nothing to do.")
        return
 
    tasks = [
        (sid, sdir, ann, args.output_dir, args.img_size, args.side_margin,
         args.top_margin, args.bottom_margin, args.threshold,
         "train" if sid in train_subjects else "test")
        for sid, sdir, ann in subjects
    ]
 
    total_rows = 0
    total_errors = 0
    label_counts = {}
    write_mode = "a" if (args.skip_existing and manifest_exists) else "w"
    with open(manifest_path, write_mode, newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
 
        if args.workers > 1:
            with Pool(args.workers) as pool:
                for idx, (rows, errors) in enumerate(pool.imap_unordered(process_subject, tasks), 1):
                    for r in rows:
                        writer.writerow(r)
                        label_counts[(r["split"], r["label"])] = label_counts.get((r["split"], r["label"]), 0) + 1
                    total_rows += len(rows)
                    total_errors += len(errors)
                    for e in errors[:5]:
                        print(f"  [warn] {e}", file=sys.stderr)
                    if idx % 5 == 0 or idx == len(tasks):
                        print(f"  {idx}/{len(tasks)} subjects done, {total_rows} images so far...")
        else:
            for idx, t in enumerate(tasks, 1):
                rows, errors = process_subject(t)
                for r in rows:
                    writer.writerow(r)
                    label_counts[(r["split"], r["label"])] = label_counts.get((r["split"], r["label"]), 0) + 1
                total_rows += len(rows)
                total_errors += len(errors)
                for e in errors[:5]:
                    print(f"  [warn] {e}", file=sys.stderr)
                if idx % 5 == 0 or idx == len(tasks):
                    print(f"  {idx}/{len(tasks)} subjects done, {total_rows} images so far...")
 
    print("\nDone.")
    print(f"Total images written: {total_rows}")
    print(f"Total per-frame errors (skipped): {total_errors}")
    print(f"Manifest: {manifest_path}")
    print("\nCounts (split, label):")
    for (split, label), n in sorted(label_counts.items()):
        print(f"  {split:7s} {label:9s} {n}")
 
 
if __name__ == "__main__":
    main()