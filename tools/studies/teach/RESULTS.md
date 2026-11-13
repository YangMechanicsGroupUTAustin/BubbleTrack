# ML assist ("Teach") study

Model layer: `src/bubbletrack/model/teach/`.  Scripts: `run_study.py` (records),
`analyze.py` + `confirm_tables.py` (tables), `common.py` (data, labels, simulated
examples), `policies.py` (offline policies).  Data: `test_data/lic`, `test_data/lic_hard`
(labels `test_data/lic/labels_edge.json`, `tests/test_sequence_labels_hard.json`) and
`example_1` (radius-only reference, every 3rd frame), read only.

```
python tools/studies/teach/run_study.py --phase all --workers 8      # ~45 min on the laptop
python tools/studies/teach/analyze.py > tables.md
```

## Protocol

* Per recording, the user teaches *n* labelled frames; every **other** labelled frame
  (with a circle) is a test frame.  Auto alone is scored on the same test frames.
* **Correct = accepted and within the label criterion** (radius within max(2 px, 5 %),
  centre within max(3 px, 15 % R); example_1 radius only).  A rejected frame shows no
  result in the app and counts as not correct.  `(Nw)` = N frames accepted but wrong.
* "good" = recordings where Auto is already good (lic 01-04, 09, 10, example_1; 7
  recordings).  "hard" = the other 10 with test frames (lic 05-08, lic_hard 01-05, 07;
  lic_hard 06 has no labelled circle, lic_hard 08 one labelled frame).
* **Verified frames** (the primary path): the label circle of a taught frame is a
  verified circle (as a manual fit or a locked frame would give); examples inside
  0.8 R = bubble, a ring and everything far = background, the wall zone unlabelled.
  example_1 uses Auto's accepted circle (it has no centre labels).
* **Strokes**: what a user paints, not regions: two crossing 5-px lines inside
  (+-0.5 R, +-0.35 R), three 50-degree arcs at 1.5 R and one long line >= 2.5 R away.
* Frame choice: *largest* (n biggest bubbles), *spread* (the biggest, then sizes down
  to the 20th percentile), *worst* (lowest Auto confidence: what Review lists first),
  *time* (evenly spaced), *suggest* (`suggest_frames` on Auto's own results: what the
  app can do without labels).
* `grid` phase: 48 configurations x 19 recordings, every candidate region and Auto run
  recorded per frame, policies evaluated offline.  `confirm` phase: the package's own
  `measure_frame` with its final defaults (half resolution, ring 1.15-2.0 R, two-pass
  refinement, guarded precedence) on 11 frame choices.

## Measurement policies

* **(i) prior**: Auto with the ML circle as prior, always.
* **(ii) contour**: ML's own outline, accepted when its confidence >= 0.3; raw P = 0.5
  outline, or *refined* to the half-contrast wall of the engine view.
* **(iii) combined** (owner's precedence): Auto (ML-guided) if accepted and on the ML
  region -> ML outline (always Review) -> no result.
* **guarded** (package default): Auto (ML-guided) -> **plain Auto if accepted** -> ML
  outline (always Review) -> no result.

### Final defaults, per recording (confirm, verified-spread-2: 2 verified frames)

| recording | Auto | combined (owner) | guarded (default) | Auto (ML-guided) ok/w/rej | plain Auto ok/w/rej | ML outline ok/w/rej | no result |
|---|---|---|---|---|---|---|---|
| lic/01 | 10/10 | 9/10 (1w) | 10/10 | 9/0/0 | 1/0/0 | 0/0/0 | 0 |
| lic/02 | 29/29 | 29/29 | 29/29 | 29/0/0 | 0/0/0 | 0/0/0 | 0 |
| lic/03 | 29/29 | 29/29 | 29/29 | 29/0/0 | 0/0/0 | 0/0/0 | 0 |
| lic/04 | 29/29 | 29/29 | 29/29 | 29/0/0 | 0/0/0 | 0/0/0 | 0 |
| lic/05 | 11/12 (1w) | 10/12 (2w) | 11/12 (1w) | 10/0/0 | 1/1/0 | 0/0/0 | 0 |
| lic/06 | 5/25 (2w) | 18/25 (3w) | 20/25 (4w) | 3/1/0 | 2/1/0 | 15/2/1 | 0 |
| lic/07 | 0/5 (1w) | 1/5 (4w) | 1/5 (4w) | 0/1/0 | 0/0/0 | 1/3/0 | 0 |
| lic/08 | 0/5 | 2/5 (1w) | 2/5 (1w) | 0/0/0 | 0/0/0 | 2/1/2 | 0 |
| lic/09 | 24/28 (2w) | 24/28 (1w) | 25/28 (2w) | 23/1/0 | 1/1/0 | 1/0/1 | 0 |
| lic/10 | 29/29 | 29/29 | 29/29 | 29/0/0 | 0/0/0 | 0/0/0 | 0 |
| lic_hard/01 | 7/30 | 29/30 (1w) | 29/30 (1w) | 7/0/0 | 0/0/0 | 22/1/0 | 0 |
| lic_hard/02 | 11/16 (5w) | 11/16 (5w) | 11/16 (5w) | 11/4/0 | 0/1/0 | 0/0/0 | 0 |
| lic_hard/03 | 0/1 | 1/1 | 1/1 | 0/0/0 | 0/0/0 | 1/0/0 | 0 |
| lic_hard/04 | 2/20 (2w) | 6/20 (9w) | 7/20 (9w) | 0/1/0 | 2/1/0 | 5/7/4 | 0 |
| lic_hard/05 | 0/19 | 16/19 (3w) | 16/19 (3w) | 0/0/0 | 0/0/0 | 16/3/0 | 0 |
| lic_hard/07 | 14/22 (1w) | 18/22 (4w) | 18/22 (4w) | 14/1/0 | 0/0/0 | 4/3/0 | 0 |
| example_1 | 28/28 | 28/28 | 28/28 | 28/0/0 | 0/0/0 | 0/0/0 | 0 |
| **good total** | **178/182 (2w)** | **177/182 (2w)** | **179/182 (2w)** | | | | |
| **hard total** | **50/155 (12w)** | **112/155 (32w)** | **116/155 (32w)** | | | | |

On every good recording guarded = or > Auto; the ML outline never fires there.  On
hard recordings it more than doubles the correct frames (50 -> 116); the extra wrong
ones (12 -> 32) are almost all ML outlines, which always go to Review.

### Final defaults, by taught frames (confirm)

| taught frames | good: Auto | good: combined | good: guarded | hard: Auto | hard: combined | hard: guarded |
|---|---|---|---|---|---|---|
| verified-largest-1 | 185/189 (2w) | 173/189 (2w) | 185/189 (2w) | 51/165 (13w) | 63/165 (36w) | 82/165 (42w) |
| verified-time-1 | 186/189 (1w) | 179/189 (3w) | 187/189 (2w) | 52/165 (14w) | 111/165 (38w) | 116/165 (41w) |
| verified-spread-2 | 178/182 (2w) | 177/182 (2w) | 179/182 (2w) | 50/155 (12w) | 112/155 (32w) | **116/155 (32w)** |
| verified-suggest-2 | 178/182 (2w) | 177/182 (2w) | 179/182 (2w) | 46/155 (12w) | 104/155 (43w) | 105/155 (44w) |
| verified-worst-2 | 180/182 (2w) | 173/182 (2w) | 180/182 (2w) | 53/155 (12w) | 89/155 (27w) | 101/155 (30w) |
| verified-spread-3 | 171/175 (2w) | 171/175 (3w) | 172/175 (2w) | 48/145 (11w) | 105/145 (35w) | 105/145 (36w) |
| verified-suggest-3 | 171/175 (2w) | 170/175 (2w) | 172/175 (2w) | 42/145 (11w) | 102/145 (33w) | 104/145 (35w) |
| strokes-largest-1 | 185/189 (2w) | 125/189 (62w) | 185/189 (4w) | 51/165 (13w) | 60/165 (33w) | 89/165 (30w) |
| strokes-spread-2 | 178/182 (2w) | 172/182 (3w) | 179/182 (2w) | 50/155 (12w) | 81/155 (53w) | 108/155 (36w) |
| strokes-suggest-2 | 178/182 (2w) | 174/182 (4w) | 179/182 (2w) | 46/155 (12w) | 73/155 (66w) | 96/155 (51w) |
| strokes-spread-3 | 171/175 (2w) | 167/175 (5w) | 172/175 (2w) | 48/145 (11w) | 81/145 (48w) | 102/145 (33w) |

**Why guarded and not the owner's exact precedence:** with unrepresentative examples
(one big bubble, or strokes) the ML region is sometimes on the wrong object; Auto
guided there fails, and the combined rule then reports the wrong ML outline in place of
a correct plain-Auto result: up to 74 of 182 frames lost on recordings where Auto is
good (grid, strokes-largest-2: 104/182 vs Auto 178/182).  Guarded never lost a correct
frame there in any of the 48 grid or 11 confirm configurations (it added 2 accepted-wrong
frames, sent to Review, when taught with strokes on the biggest bubbles only) and is
also best on the hard recordings.  `MeasureConfig(policy=
"combined")` gives the owner's exact precedence.

### Policies on the grid records (verified-spread-2, first refinement setting)

| | Auto | (i) prior | (ii) raw P=0.5 | (ii) refined | (iii) combined | guarded |
|---|---|---|---|---|---|---|
| good | 178/182 (2w) | 178/182 (2w) | 126/182 (53w) | 173/182 (6w) | 178/182 (3w) | 179/182 (2w) |
| hard | 50/155 (12w) | 51/155 (12w) | 81/155 (64w) | 113/155 (32w) | 112/155 (34w) | 114/155 (35w) |

* (i) alone changes nothing: where Auto fails, it fails to *measure* a wall even inside
  the right region (blood, moving speckle, low contrast).  The gain comes from ML's own
  outline.
* The raw P = 0.5 outline is biased outward (median +4 % to +13 % R: the classifier's
  boundary falls inside the unlabelled wall zone).  Moving it to the half-contrast wall
  of the engine view is essential (81 -> 113 hard, 126 -> 173 good).

## Refinement of ML's outline (all 30 main grid configurations, contour policy)

| | raw P=0.5 | +-0.15 R, 8 sectors | 1 sector | 4 sectors | +-0.10 R | +-0.25 R | **two passes (0.2 R, 0.1 R)** |
|---|---|---|---|---|---|---|---|
| good | 3209/5278 (1705w) | 4148 (766w) | 4166 (748w) | 4153 (761w) | 3991 (923w) | 4152 (762w) | **4362 (552w)** |
| hard | 1281/4408 (2206w) | 2294 (1193w) | 2262 (1225w) | 2306 (1181w) | 2185 (1302w) | 2183 (1304w) | **2505 (982w)** |

The view is averaged along ML's outline (mean profile vs signed distance, per angular
sector, integral crossing estimator).  Per-pixel thresholding was pushed outward by
the skewed ratio noise of moving speckle (lic 09: +8 % bias).

## Number and choice of taught frames (grid, guarded; Auto on the same frames)

| examples | choice | n=1 hard | n=2 hard | n=3 hard | n=5 hard | good (any n) |
|---|---|---|---|---|---|---|
| verified | largest | 69/165 (Auto 51) | 80/155 (48) | 66/145 (45) | 68/127 (37) | = Auto |
| verified | spread | (= largest) | **114/155 (50)** | 108/145 (48) | 92/127 (42) | Auto +1 |
| verified | worst | 83/165 (55) | 89/155 (53) | 80/145 (52) | 83/127 (48) | = Auto |
| verified | time | **115/165 (52)** | 108/156 (49) | 100/145 (45) | 89/127 (40) | = Auto (+1) |
| strokes | largest | 90/165 (51) | 97/155 (48) | 91/145 (45) | 86/127 (37) | = Auto |
| strokes | spread | (= largest) | 106/155 (50) | 99/145 (48) | 90/127 (42) | Auto +1 |
| strokes | worst | 91/165 (55) | 93/155 (53) | 92/145 (52) | 97/127 (48) | = Auto |
| strokes | time | 100/165 (52) | 99/156 (49) | 94/145 (45) | 90/127 (40) | Auto +0/+1 |

* **Two verified frames of different sizes** (the biggest bubble and a small one) are
  best; more frames do not help (the test sets shrink, but the fraction correct also
  stops rising: 74 % at n=2, 74 % at n=3, 72 % at n=5).
* **Never only the biggest bubble**: a model taught on Rmax alone misses the small
  bubbles (69/165).  One mid-recording frame (time-1) is the best single frame.
* Verified frames beat strokes at equal frames (114 vs 106 hard), with no painting.
  Strokes help most where Auto is worst (worst-5: 97/127); they remain useful to add
  on the frames where ML is wrong.
* `suggest_frames` (sizes from Auto's own accepted results, no labels) gets 105/155 vs
  116/155 with label sizes: on hard recordings Auto accepts few frames, so its sizes
  are a weaker guide; still +59 over Auto on the same frames (105 vs 46).

## Features, half resolution, classifier, margins (grid, verified-spread-2)

| variant | good: guarded | hard: guarded | hard: ML outline alone |
|---|---|---|---|
| raw + view (default) | 179/182 (2w) | 114/155 (35w) | 113/155 (32w) |
| raw only | 179/182 (2w) | 100/155 (44w) | 92/155 (44w) |
| view only | 178/182 (3w) | 95/155 (36w) | 86/155 (38w) |
| raw + view, half resolution | 179/182 (2w) | 114/155 (35w) | 113/155 (32w) |
| RandomForest instead of ExtraTrees | 179/182 (2w) | 114/155 (34w) | 110/155 (33w) |
| inner 0.6 R | 179/182 (3w) | 107/155 (37w) | 85/155 (47w) |
| inner 0.9 R | 179/182 (2w) | 103/155 (43w) | 100/155 (43w) |
| ring 1.15-2.0 R | 179/182 (2w) | 118/155 (29w) | 119/155 (25w) |
| ring 1.5-2.5 R, far 3 R | 179/182 (2w) | 103/155 (43w) | 101/155 (41w) |
| no far background | 179/182 (2w) | 112/155 (35w) | 113/155 (32w) |

* **The reference channel helps**: the engine view (frame / reference, or the
  reference-free surface normalisation) adds 14 hard frames over the raw frame alone;
  both channels together beat either.
* **Half resolution costs nothing** (same counts, also with strokes: 106 vs 106) and
  predicts 4x fewer pixels: default.
* ExtraTrees >= RandomForest (equal with verified frames, 8 more correct ML outlines on
  good recordings with strokes) and trains several times faster (probe on 20 k pixels:
  0.6 s vs 4.3 s single-threaded): default.
* Margins: inner 0.8 R; ring 1.15-2.0 R was best on this configuration; over the 7
  verified confirm configurations it is equal to 1.25-2.0 R (729 vs 733 correct, 260 vs
  258 wrong), and is kept with half resolution as the confirmed default.

## Confidence of the ML outline

Confidence = separation (mean P inside - mean P 0.3-0.6 R outside) x shape (solidity)
x clutter (other P >= 0.5 area within R).  On frames where guarded falls back to the ML
outline: AUC 0.80 (correct median 0.89, wrong 0.56); best single terms separation and
circularity (0.79), p_in (0.77); the P-map "sharpness" is useless with verified-frame
examples (P is undecided across the unlabelled wall zone by construction).

| gate | good: ML outline correct / wrong | hard: ML outline correct / wrong |
|---|---|---|
| 0.0 | 10 / 34 | 1336 / 1204 |
| 0.3 (default) | 10 / 10 | 1329 / 766 |
| 0.5 | 10 / 10 | 1286 / 663 |
| 0.6 | 9 / 10 | 1240 / 566 |

(guarded, all 30 main grid configurations.)  The gate removes 36 % of the wrong ML
outlines at a cost of 0.5 % of the correct ones; the rest goes to Review anyway.

## Novelty ("does this frame look like the training frames?")

Per feature, the mean distance between the frame's quantiles and the training
quantiles over the training spread; median over features.  A model trained on 3
verified frames of one recording:

| warn above | frames of the same recording flagged | frames of the 18 other recordings flagged |
|---|---|---|
| 0.05 | 10 % | 95 % |
| **0.10 (default)** | **0 %** | **64 %** |
| 0.15 | 0 % | 52 % |
| 0.30 | 0 % | 41 % |

Same recording: median 0.025, 95th percentile 0.051; other recordings: median 0.168.

## Timing

One process, nothing else running, 3 recordings (lic 06, lic_hard 05, lic 09), 6 test
frames each; 400 x 250 frames, full-frame ROI.  Train = slowest of the 3 recordings
(frames read from disk, features, fit); predict = features + forest, median, 4 threads /
1 thread; measure_frame = the whole default pipeline (predict, engine view, region, two-
pass refinement, guided Auto and plain Auto), median.  Plain Auto alone: 64 ms.

| examples | frames | train ms (half res) | predict ms (half res, 4 / 1 threads) | measure_frame ms (half res) | train ms (full res) | predict ms (full res, 4 / 1) | measure_frame ms (full res) | training pixels |
|---|---|---|---|---|---|---|---|---|
| verified | 1 | 190 | 95 / 81 | 309 | 300 | 237 / 339 | 439 | 6000 |
| verified | 2 | 255 | 96 / 86 | 310 | 476 | 174 / 317 | 377 | 10364 |
| verified | 3 | 347 | 76 / 92 | 273 | 609 | 200 / 324 | 391 | 14865 |
| verified | 5 | 462 | 85 / 97 | 267 | 820 | 197 / 376 | 387 | 24981 |
| strokes | 1 | 137 | 82 / 75 | 256 | 254 | 174 / 271 | 350 | 2121 |
| strokes | 2 | 191 | 80 / 80 | 266 | 334 | 182 / 317 | 364 | 3930 |
| strokes | 3 | 299 | 79 / 83 | 274 | 489 | 183 / 307 | 379 | 5553 |
| strokes | 5 | 449 | 80 / 91 | 282 | 724 | 193 / 316 | 383 | 9016 |

* **Training < 0.5 s** for 1-5 verified frames at half resolution (< 1.2 s at full
  resolution); the brief's budget is 2 s.
* **Prediction 70-115 ms per frame at half resolution** (budget 300 ms); full
  resolution needs 170-430 ms.  Threads hardly matter at half resolution.
* A whole ML-assisted frame is ~0.25-0.35 s (two engine runs included; ~0.07 s less
  when the Fit All result is passed as ``auto``): a 256-frame recording in ~1-1.5 min on
  one core, or proportionally less with the batch workers.
* Prediction is bit-for-bit deterministic for any thread count (row chunks, each
  predicted by a single-threaded forest); training is deterministic by its fixed seed.

## Dependency

Training needs scikit-learn (imported lazily; `MLUnavailableError` without it).  It
is not a project dependency yet.  Adding `scikit-learn>=1.3` brings scikit-learn
(36 MB installed without its tests, 48 MB with them), joblib (2.4 MB) and threadpoolctl
(numpy and scipy are already dependencies); the PyInstaller bundle would grow by roughly
the size of the compiled sklearn modules actually imported (`sklearn.ensemble`,
`sklearn.tree`, `sklearn.utils`: expect 15-30 MB).  The model tests skip without it, so
CI (which does not install it) runs only the pure-numpy tests until it is added.
