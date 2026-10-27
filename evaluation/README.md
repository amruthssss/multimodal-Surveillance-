# Labeled evaluation set

Copy `manifest.example.csv` to `manifest.csv`, add locally sourced clips, and set
each `label` to the primary event actually present in that clip. Include normal
clips as well as event clips; normal examples are necessary to measure false
alarms. Keep source media local and do not commit private or sensitive footage.

Supported labels are `normal`, `explosion`, `fire`, `smoke`, `accident`, and
`gunshot`. Video evaluates the visual pipeline; audio evaluates YAMNet's
gunshot/explosion classifier using the same two-window trigger as live audio
monitoring. Gunshots must use audio clips. Audio clips can be WAV or another
format supported by SoundFile. Each clip should contain one primary label. Paths
are relative to the manifest file unless absolute.

Run from the repository root:

```powershell
python scripts/evaluate_labeled_videos.py --manifest evaluation/manifest.csv --output evaluation/report.json
```

The report includes clip-level accuracy and per-event precision, recall, and F1.
Use separate clips for tuning and final evaluation so threshold adjustments are
not measured on the same samples that informed them. Metrics show `null` where
there are no reviewed positives or negatives for that measure.

Live operator labels are stored in the configured SQLAlchemy database table
`detection_reviews`; query the dashboard's Event Intelligence tab or the
`/api/detections/reviews` and `/api/detections/evaluation` endpoints to inspect
the reviewed examples and their aggregate metrics.
