# ImgArchive Studio

Custom image archive builder, viewer, and extractor for large image collections.

ImgArchive Studio creates a custom .iarc archive format optimized for image folders where storage size, random access, and app-level control matter more than generic ZIP/RAR workflows.
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/fc1259b3-c3ca-4574-b6b1-9d3c6e122352" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/990c82ad-0ed5-418e-bc70-93f7f20ea3d5" />

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/4a60706c-47a9-4423-bccf-cce0fec7fff0" />


It supports:
- sequence-style archives for video frames or near-sequential images
- gallery-style archives for mixed image sets grouped by face identity
- random-access listing and extraction
- on-the-fly frame reconstruction using RIFE
- archive maintenance tools like compact and repair
- export to video with optional audio using ffmpeg
- project/session files via .ias

---

## Highlights

- Custom archive format: .iarc
- Project/session format: .ias
- Full desktop GUI built with Tkinter
- GPU-first workflow for heavy operations
- RIFE interpolation for reconstructed frames
- RAFT / face / body analysis support for smarter compression planning
- Random access to individual frames without unpacking the whole archive
- Archive actions: list, extract, delete, restore, compact, repair, split
- Export archive to video with optional user-selected audio track
- Generate contact sheets inside the app
- Handles large frame counts far beyond what is comfortable in flat folders

---

## Why this exists

Traditional archive tools like ZIP, RAR, and 7z treat images as generic files. They do not understand:
- visual similarity between frames
- face/body safety concerns
- random-access decode requirements for apps
- archive-level frame operations like delete/restore/compact/repair

ImgArchive Studio is designed specifically for image archives that need to behave more like an indexed media container than a generic compressed folder.

---

## Archive modes

### 1. Sequence Mode
Use this for:
- video frames
- animation frames
- captures where neighboring images are visually close

In this mode, the builder assumes temporal continuity. Intermediate frames may be reconstructed from nearby keyframes using RIFE.

### 2. Gallery Mode
Use this for:
- mixed image folders
- multiple images per person
- non-video collections where images are not sequential

In this mode, the builder groups images by face identity, sorts visually similar images together within a person cluster, and only interpolates when images are extremely close. Images with no face are stored conservatively as keyframes.

Important: Gallery mode is intentionally more conservative than sequence mode, because RIFE will create ghosted blends if two different images are forced into interpolation.

---

## File formats

### .iarc
Main archive format.

Stores:
- archive header
- indexed frame table
- keyframe blobs
- optional residual blobs
- frame dependency metadata
- archive checksum footer

### .ias
Project/session file.

Stores:
- source folder path
- archive path
- optional audio file path for video export
- build settings snapshot
- export settings
- contact sheet settings
- notes/bookmarks/tags

One .ias is intended to map to one .iarc archive.

### Split archives
When splitting an archive, output files are named like:
- name.iar001
- name.iar002
- name.iar003

Up to 999 parts are supported.

---

## Compression model

At a high level, ImgArchive Studio uses a hybrid approach:

- some frames are stored directly as compressed WebP keyframes
- some frames are reconstructed from neighboring keyframes using RIFE
- some reconstructed frames also store a residual patch for correction

Frame types:

- K = Keyframe
- I = Interpolated
- R = Interpolated + Residual
- C = Forced Keyframe
- D = Deleted

### Why archives can be much smaller than ZIP/RAR/7z

If your source folder is mostly PNG/BMP/TIFF or otherwise poorly suited to generic file archiving, converting selected frames to WebP lossy alone can produce major savings.

That means even before interpolation savings, .iarc may already be dramatically smaller than standard archive tools.

---

## Main features

### Builder
- scan large image folders
- sequence analysis with progress, ETA, throughput
- sequence mode and gallery mode
- face-safe/body-safe build planning
- residual-aware storage planning
- fast analyzer for large folders
- project auto-save to .ias
- batch archive builder for multiple folders

### Viewer / Extractor
- open archive instantly via index
- virtualized frame listing for large archives
- decode selected frame on demand
- extract selected / range / all / keyframes only
- delete / restore frames
- toolbar frame counter
- recent files menu
- keyboard shortcuts

### Archive maintenance
- verify archive
- compact archive
- repair archive
- split archive by frame count or GOP boundaries

### Export / review
- export archive to video
- optional audio mux using user-selected audio file
- contact sheet generation inside app
- source/decoded review workflow via project state

---

## Requirements

### Core
- Python 3.11+
- Tkinter
- NumPy
- OpenCV
- Pillow
- PyTorch with CUDA recommended

### Recommended / optional
- insightface
- onnxruntime-gpu
- ffmpeg available in system PATH

### Expected environment
This project was built around a GPU-enabled Python environment with CUDA and image/vision tooling already installed.

---

## External models

The app can make use of the following model families depending on the features enabled:

- RIFE for interpolation
- RAFT for motion analysis
- InsightFace / ArcFace for face detection and identity consistency
- YOLO segmentation / pose models for body-aware analysis
- DeepLab human segmentation fallback
- Depth Anything for optional depth-aware analysis
- RealESRGAN / SwinIR for optional upscaling
- GFPGAN for optional face enhancement
- TransNetV2 for scene cut detection

Model paths are configured in the Settings tab.

---

## ffmpeg

Video export uses ffmpeg through subprocess calls.

Make sure this works in a terminal:

ffmpeg -version

If not, add ffmpeg to your PATH.

---

## Running the app

Run the main application:

python iarc.py

If you are using the standalone viewer/extractor:

python iarc_viewer.py

---

## Typical workflow

### Create an archive
1. Open the app
2. Go to Archive Builder
3. Select a source folder
4. Scan the folder
5. Choose Sequence Mode or Gallery Mode
6. Adjust quality / GOP / thresholds / safety toggles
7. Run analysis
8. Review estimated keyframes / interpolated frames / savings
9. Build the archive
10. A matching .ias project file is auto-created beside the archive

### View and extract
1. Open the archive in Archive Viewer
2. Browse the frame list
3. Decode selected frames on demand
4. Extract selected / range / all as needed

### Export to video
1. Open an archive
2. Go to Export -> Export to Video
3. Choose output format and FPS
4. Optionally choose an audio file
5. Export through ffmpeg

### Contact sheet
1. Open an archive
2. Go to Export -> Generate Contact Sheet
3. Configure every-Nth-frame and thumbnail size
4. Generate
5. Review in preview window and save if desired

---

## Project files (.ias)

Project files are JSON-based session files.

They allow the app to remember:
- where the source images came from
- where the archive lives
- what audio file should be used for export
- the build mode and thresholds
- contact sheet and export preferences

When an archive is opened, the app looks for a matching .ias beside it.

---

## Archive maintenance

### Verify
Checks archive integrity and reports obvious missing/corrupt stored data.

### Compact
Rebuilds a clean archive after logical deletions.

Use this when:
- you deleted frames
- you want to physically remove dead data
- you want a smaller final archive

### Repair
Attempts to recover from:
- corrupted keyframe blobs
- corrupted residual blobs
- orphaned interpolated frames
- broken parent references

Repair is conservative and may promote damaged frames to keyframes or demote broken residual frames to simpler forms.

---

## Keyboard shortcuts

- Ctrl+O — Open archive
- Ctrl+N — New archive from folder
- Ctrl+S — Save project
- Ctrl+Shift+O — Open project
- Ctrl+Shift+V — Export to video
- Ctrl+Shift+C — Generate contact sheet
- Ctrl+E — Extract selected frame
- Left — Previous frame
- Right — Next frame
- Space — Decode current frame
- Home — First frame
- End — Last frame
- Delete — Delete selected frame
- F1 — Help
- F5 — Reload archive
- F11 — Toggle fullscreen

---

## When to use which mode

### Use Sequence Mode when:
- images are ordered in time
- neighboring frames are visually close
- you want the strongest RIFE-based savings

### Use Gallery Mode when:
- the folder contains many different images per person
- the images are not video frames
- face identity grouping makes more sense than timeline order

### Do not expect Gallery Mode to behave like video interpolation
Even if two images have the same person, if they are different shots, poses, or backgrounds, interpolating between them may create blended/ghosted results.

Gallery Mode intentionally uses stricter thresholds and more keyframes for this reason.

---

## Known limitations

- This is a custom archive format, not a standard media container
- Lossy compression is part of the design
- Sequence Mode assumes neighbor similarity
- Gallery Mode is helpful but cannot magically interpolate unrelated photos cleanly
- If source images have mixed aspect ratios and wildly different dimensions, conservative storage is usually better than aggressive interpolation
- Some advanced model combinations may require environment-specific tuning

---

## Repository structure

Typical files:

- iarc.py — main application
- iarc_viewer.py — standalone viewer/extractor API + GUI
- HELP.LLM — in-app help content
- imgarchive_settings.json — global app settings
- *.ias — per-archive project/session files

---

## Example use cases

- long frame sequences extracted from generated video
- animation frame archives
- AI workflow asset bundles
- same-person image collections grouped by identity
- app-side indexed archives where listing/extraction must be scriptable

---

## Standalone viewer / extractor API

The standalone viewer supports both GUI and programmatic use.

Example:

python
from iarc_viewer import IARCReader

with IARCReader("archive.iarc") as reader:
 info = reader.info()
 img = reader.get_frame(42)
 reader.extract_frame(42, "frame42.png")


If you do not want markdown fences inside the README, remove the example block above after generating the file.

---

## Notes on quality vs size

This project can often beat ZIP/RAR/7z on image folders because it is not just compressing files byte-for-byte. It is also:
- converting selected frames to WebP lossy
- optionally reducing stored frame count
- storing only keyframes/residuals where useful

That means a .iarc archive may be dramatically smaller than generic archives even when interpolation savings are modest.

---

## Roadmap / ideas

Potential future directions:
- stronger gallery-mode clustering and ordering
- richer compare tools for source vs decoded frames
- archive metadata tagging and search
- better standalone repair in the external viewer
- more export presets

---

## Disclaimer

This project is specialized and opinionated. It is built for image-heavy workflows where:
- you control the environment
- GPU acceleration is available
- custom archive behavior is more important than compatibility with generic tools

Always keep original sources when testing new archive settings.

---

## License

Add your preferred license here.

---

## Author note

Built as a custom Python image archive workflow with focus on large collections, GUI tooling, and app-callable extraction.
