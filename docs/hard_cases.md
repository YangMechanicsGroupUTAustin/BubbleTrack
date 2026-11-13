# Hard cases: frames BubbleTrack gets wrong

A *hard case* is a short stretch of a recording where the app measures the
bubble wrongly, saved together with **your fixes**. The fixes are the right
answers ("labels"). Every hard case the group sends becomes a regression
test: before a new version is released, the maintainer re-runs it on all of
them and sees at once what got better and what got worse.

## For group members: send a hard case

### 1. Fix the frames first

Only frames you fixed are scored. A fix is one of:

| What you do | Where | What it tells the maintainer |
|---|---|---|
| Fix a frame (drag a circle, or click edge points) | Review > Fix (F), or chart: right-click a point > Fix frame N… | this circle is right |
| Accept / lock a result or Auto's guess | Review > Accept (A) / Accept all (Shift+A), or chart: Shift+drag a range > right-click > Lock | this result is right |
| Mark frames as no bubble | Review > No bubble here (N) / No bubble on all (Shift+N), or chart: Shift+drag a range > right-click > Mark as no bubble | there is no bubble here |

**Only "No bubble" says a frame is empty.** Accept never does: a frame
without a result is often a bubble Auto missed, so Accept leaves it alone
and it is sent *unlabelled* until you fix it or mark it "no bubble".

Frames you did not touch are sent as well (they show the context) but count
as *unlabelled*. Fix a few frames where it goes wrong and, ideally, a few
where it is right just before and after.

### 2. Export

Choose the frames, then either

- **File > Export Hard Case...** - uses the range you brushed on the R(t)
  chart (Shift+drag); without one, the selected Review row; else the frame
  on screen. You can change the range in the dialog; or
- right-click a row in the **Review** dock > **Export as hard case...**

In the dialog, write what is wrong ("What is wrong here?", e.g. *the radius
jumps when the bubble touches the wall*) and, if you like, your name (it is
remembered). The dialog shows how many frames, the size and how many of them
you fixed. Click **Save...** and pick a place for the `.zip`.

### 3. Send the .zip

Send the file any way you like: Box, Teams, e-mail, USB stick. Renaming it
is fine.

**What is inside:** the frames you chose plus 3 neighbours on each side, and
in Auto mode the reference frame, all **cropped to the ROI** and stored
losslessly at their original bit depth (videos and `.dat` recordings work
too: the decoded frames are stored); the settings (detection mode and all
parameters); and per frame what the app showed and what you fixed. The
recording is identified by its file or folder **name only**, never by its
full path.

**Size:** about 70-110 KB per frame for a 250 x 400 px, 16-bit ROI; a typical
case of 20 frames is 2-3 MB. The app warns above 50 MB: choose a shorter
range or a smaller ROI then.

## For the maintainer: run the library

Put the bundles in one folder (subfolders are fine, and so are bundles that
were unpacked; re-zipped ones too). Then

```
bubbletrack-cli eval-cases path/to/cases --out reports
```

re-runs the current engine on every bundle with the bundle's own parameters
and reference frame, and writes `reports/hard_cases_report.md` and
`reports/hard_cases_report.json`. Keep the JSON of a release and compare later
runs against it:

```
bubbletrack-cli eval-cases path/to/cases --out reports/new --baseline reports/v3.1.json
```

| Option | Meaning |
|---|---|
| `--baseline REPORT.json` | an earlier report; frames correct there and not now are *regressions* |
| `--out DIR`, `--name STEM` | where the report goes (default: current folder, `hard_cases_report`) |
| `--radius-tol-px`, `--radius-tol-frac` | radius tolerance, default `max(1 px, 3 %)` of the label's R |
| `--center-tol-px`, `--center-tol-frac` | centre tolerance, default `max(2 px, 10 % R)` |
| `--no-tracking` | Auto mode: no tracking across frames, even where the user had it on |
| `-v` | debug logging |

Exit status: **0** no regressions, **1** regressions against the baseline
(use it in CI or a release checklist), **2** bad arguments, unreadable
baseline, no bundles found or the report cannot be written. A bundle that cannot be read or replayed is
listed as an error and does not stop the run; against a baseline its frames
that were correct count as regressions.

### Reading the report

Each labelled frame is one of

- **correct** - an *accepted* result within the tolerance of the label; for
  a "no bubble" label: no result. A rejected frame shows no result in the
  app, so it is never correct for a bubble label;
- **wrong** - an accepted result that does not match, or any result on a "no
  bubble" frame. The worst kind: the app shows a wrong number;
- **rejected** - the label has a bubble, the app shows a gap.

The *Summary* table has one row per case. **Reproduced** counts the app's
automatic results at export (Fit All or tracking) that this run gives again
within the tolerance: it drops when the engine changed on that case, so
`17/17` means "same answers as the user saw". The *Against baseline* section
lists regressions (bold) and fixed frames; *Cases* shows the note of each
case and every frame that is not correct, with the label's radius, the
radius now and both errors.

### How the replay works, and its limits

- The frames are ROI crops, so the ROI becomes the whole crop and results are
  shifted back by the ROI offset. Auto mode works on the ROI only and gives
  the same numbers as in the app. The threshold-based modes (Intensity,
  Gradient, Area) normalise with whole-frame statistics, so a crop can come
  out slightly differently; *Reproduced* shows how much.
- Auto mode uses the bundle's reference frame (chosen or set when it was
  exported) and the seed click. Tracking across frames is replayed over the
  stored frames only (not the whole recording) and never uses the labels as
  anchors.
- Only labelled frames are scored; unlabelled frames still count in
  *Reproduced*.

### Bundle format

```
case.json                   format "bubbletrack-hard-case", version 1
frames/frame_000121.tif     ROI crop of the app's frame 121 (uint8/uint16/float, lossless)
```

`case.json` holds `case_id` (unique per export; reports and baselines match
cases by it), `app_version`, `engine_version`, `reporter`, `note`, `source`
(kind, name, frame count, frame size), `roi` (`gridx`, `gridy`, 1-based
inclusive, and `offset`), `detection` (`mode`, `params` = the session
parameters, `tracking`, `reference_index`, `max_radius`), `selection` and per
frame its 0-based `index`, `role` (selected / context / reference), the
app's `result` (radius, centre, confidence, flags, source, locked) and its
`label` (`bubble` with radius and centre, `no_bubble`, or `unlabelled`, and
`by`: manual / accepted). Radii and centres are in pixels of the
full frame, 1-based, as in the app.
