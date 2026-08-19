"""
ZIP FILE SOURCE: https://www.kaggle.com/datasets/dhruv413/gaze-capture-20gb-zip

GazeCapture 3-class (left/straight/right) pipeline - standalone, from scratch
================================================================================
 
Goes directly from the raw unzipped GazeCapture dump to 224x224 face crops
labeled left/straight/right, with no dependency on any other script or
pre-built manifest. Only the horizontal gaze signal is used - vertical
(up/down) gaze is ignored entirely, so a frame where someone was looking
down-and-right still gets labeled "right", not dropped or reclassified.
 
Cropping:
GazeCapture ships Apple Vision's per-frame face bounding box in
appleFace.json (X, Y, W, H, IsValid), so we crop directly from that instead
of running a face detector, with a small added margin for context.
 
Labeling:
dotInfo.json's XCam is the on-screen dot's position in centimeters, relative
to the camera lens - not something this script computes, it's ground truth
already provided by GazeCapture's own authors (converted by them from pixel
position using each device's screen size and camera mounting geometry).
Since the subject was told to stare at that dot, XCam is a direct proxy for
horizontal gaze relative to the camera:
 
    |XCam| <= --threshold cm  -> straight
    XCam >  threshold         -> right
    XCam < -threshold         -> left
 
Caveat: XCam is a linear distance, not an angle, and GazeCapture doesn't
record how far the subject's face was from the camera in a given frame. The
same cm offset implies a bigger eye-rotation angle for someone holding the
phone close to their face than at arm's length - this script does not
correct for that, so --threshold is a fixed physical tolerance, not a
calibrated visual angle.
 
Optional quality filters (both opt-in, off by default - see the docstrings
inline below for exactly how each works):
    --require_eye_valid   also require Apple's eye detections to be valid
    --filter_motion        drop frames captured while the phone moved/rotated
                            too fast (uses motion.json, matched to each frame
                            by DotNum + nearest timestamp)
 
Split: subject-level 80/20 - each subject assigned entirely to train or test
(never split across both), so a person's face never leaks across the split.
 
Expected input layout (per subject):
    <input_dir>/<subjectID>/<subjectID>/
        info.json, frames.json, appleFace.json, appleLeftEye.json,
        appleRightEye.json, dotInfo.json, motion.json, frames/*.jpg
 
Output layout:
    <output_dir>/train/<label>/<subjectID>_<frameID>.jpg
    <output_dir>/test/<label>/<subjectID>_<frameID>.jpg
    <output_dir>/manifest.csv
 
Usage:
    python3 gazecapture_3class.py --input_dir /path/to/GazeCapture --output_dir /path/to/out
    python3 gazecapture_3class.py --input_dir ... --output_dir ... --limit_subjects 10  # dry run
"""
 
import argparse
import csv
import json
import math
import os
import random
import sys
from multiprocessing import Pool, cpu_count
 
from PIL import Image
 
LABELS = ["left", "straight", "right"]
 
 
def classify_horizontal(x_cam, threshold):
    if abs(x_cam) <= threshold:
        return "straight"
    return "right" if x_cam > 0 else "left"
 
 
def find_subject_dirs(input_dir):
    """GazeCapture subjects are stored as <id>/<id>/ (nested duplicate name)."""
    subjects = []
    for entry in sorted(os.listdir(input_dir)):
        outer = os.path.join(input_dir, entry)
        if not os.path.isdir(outer):
            continue
        inner = os.path.join(outer, entry)
        if os.path.isdir(inner) and os.path.exists(os.path.join(inner, "info.json")):
            subjects.append((entry, inner))
        elif os.path.exists(os.path.join(outer, "info.json")):
            subjects.append((entry, outer))
    return subjects
 
 
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
 
 
def vec_mag(v):
    return math.sqrt(v["X"] ** 2 + v["Y"] ** 2 + v["Z"] ** 2)
 
 
def build_motion_lookup(motion):
    """DotNum -> sorted list of (Time, accel_mag, rot_mag). motion.json is a
    separate, higher-frequency sensor stream, not 1:1 with frames, and its
    Time resets per dot (same convention as dotInfo.json's Time)."""
    lookup = {}
    for m in motion:
        try:
            d = m["DotNum"]
            t = m["Time"]
            accel = vec_mag(m["UserAcceleration"])
            rot = vec_mag(m["RotationRate"])
        except (KeyError, TypeError):
            continue
        lookup.setdefault(d, []).append((t, accel, rot))
    for d in lookup:
        lookup[d].sort(key=lambda e: e[0])
    return lookup
 
 
def nearest_motion(lookup, dotnum, t):
    samples = lookup.get(dotnum)
    if not samples:
        return None
    best = min(samples, key=lambda e: abs(e[0] - t))
    return best[1], best[2]
 
 
def process_subject(args):
    (subject_id, subject_dir, output_dir, img_size, margin, threshold,
     require_eye_valid, filter_motion, max_accel, max_rotation, split) = args
    rows = []
    errors = []
    try:
        info = load_json(os.path.join(subject_dir, "info.json"))
        frames = load_json(os.path.join(subject_dir, "frames.json"))
        face = load_json(os.path.join(subject_dir, "appleFace.json"))
        dot = load_json(os.path.join(subject_dir, "dotInfo.json"))
    except Exception as e:
        return [], [f"{subject_id}: failed to load metadata ({e})"]
 
    left_eye = right_eye = None
    if require_eye_valid:
        try:
            left_eye = load_json(os.path.join(subject_dir, "appleLeftEye.json"))
            right_eye = load_json(os.path.join(subject_dir, "appleRightEye.json"))
        except Exception as e:
            errors.append(f"{subject_id}: --require_eye_valid set but eye files missing ({e}); not filtering on eyes for this subject")
            require_eye_valid = False
 
    motion_lookup = None
    if filter_motion:
        try:
            motion = load_json(os.path.join(subject_dir, "motion.json"))
            motion_lookup = build_motion_lookup(motion)
        except Exception as e:
            errors.append(f"{subject_id}: --filter_motion set but motion.json missing ({e}); not filtering on motion for this subject")
 
    frames_dir = os.path.join(subject_dir, "frames")
    n = len(frames)
    for i in range(n):
        try:
            if i >= len(face["IsValid"]) or not face["IsValid"][i]:
                continue
 
            if require_eye_valid and left_eye is not None and right_eye is not None:
                lv = left_eye["IsValid"][i] if i < len(left_eye["IsValid"]) else 0
                rv = right_eye["IsValid"][i] if i < len(right_eye["IsValid"]) else 0
                if not (lv and rv):
                    continue
 
            if motion_lookup is not None:
                m = nearest_motion(motion_lookup, dot["DotNum"][i], dot["Time"][i])
                if m is not None:
                    accel_mag, rot_mag = m
                    if accel_mag > max_accel or rot_mag > max_rotation:
                        continue
 
            frame_name = frames[i]
            src_path = os.path.join(frames_dir, frame_name)
            if not os.path.exists(src_path):
                continue
 
            x, y, w, h = face["X"][i], face["Y"][i], face["W"][i], face["H"][i]
            x_cam = dot["XCam"][i]
            label = classify_horizontal(x_cam, threshold)
 
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                img_w, img_h = im.size
                mx, my = w * margin, h * margin
                left = max(0, x - mx)
                top = max(0, y - my)
                right = min(img_w, x + w + mx)
                bottom = min(img_h, y + h + my)
                if right <= left or bottom <= top:
                    continue
                crop = im.crop((left, top, right, bottom)).resize((img_size, img_size), Image.LANCZOS)
 
                out_dir = os.path.join(output_dir, split, label)
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"{subject_id}_{os.path.splitext(frame_name)[0]}.jpg"
                out_path = os.path.join(out_dir, out_name)
                crop.save(out_path, "JPEG", quality=95)
 
            rows.append({
                "subject": subject_id,
                "frame": frame_name,
                "split": split,
                "label": label,
                "x_cam": x_cam,
                "out_path": os.path.relpath(out_path, output_dir),
            })
        except Exception as e:
            errors.append(f"{subject_id}/{frames[i] if i < len(frames) else i}: {e}")
 
    return rows, errors
 
 
def split_subjects(subjects, train_frac, seed):
    subjects = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    cut = int(round(len(subjects) * train_frac))
    return set(subjects[:cut]), set(subjects[cut:])
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="Path to unzipped GazeCapture folder")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--margin", type=float, default=0.15, help="Extra context around face bbox, as fraction of box size")
    ap.add_argument("--threshold", type=float, default=3.0, help="cm, default 3.0")
    ap.add_argument("--require_eye_valid", action="store_true")
    ap.add_argument("--filter_motion", action="store_true")
    ap.add_argument("--max_accel", type=float, default=0.08, help="g, used only if --filter_motion")
    ap.add_argument("--max_rotation", type=float, default=0.3, help="rad/s, used only if --filter_motion")
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
    fieldnames = ["subject", "frame", "split", "label", "x_cam", "out_path"]
    manifest_exists = os.path.exists(manifest_path)
    already_done = set()
    if args.skip_existing and manifest_exists:
        with open(manifest_path, "r", newline="") as mf:
            for row in csv.DictReader(mf):
                already_done.add(row["subject"])
        subjects = [(sid, sdir) for sid, sdir in subjects if sid not in already_done]
 
    print(f"Subjects total: {len(all_ids)} ({len(train_subjects)} train / {len(test_subjects)} test). "
          f"Processing {len(subjects)} this run (already_done_skipped={len(already_done)}), {args.workers} workers, threshold={args.threshold}cm")
 
    if not subjects:
        print("Nothing to do.")
        return
 
    tasks = [
        (sid, sdir, args.output_dir, args.img_size, args.margin, args.threshold,
         args.require_eye_valid, args.filter_motion, args.max_accel, args.max_rotation,
         "train" if sid in train_subjects else "test")
        for sid, sdir in subjects
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
                    if idx % 10 == 0 or idx == len(tasks):
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
                if idx % 10 == 0 or idx == len(tasks):
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