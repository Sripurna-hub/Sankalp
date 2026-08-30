# Autonomous Disaster Response Engine

A Streamlit platform for citizen-sourced disaster reporting, drone-geofence verification, and AI-driven infrastructure damage assessment — built to help rescue teams triage and prioritize response during active disaster events.

## Overview

Citizens and field teams upload photos/videos of disaster-affected areas. Each upload is checked against an active drone reconnaissance zone (geofence) and, once verified, run through a local computer-vision pipeline that detects infrastructure, classifies damage severity, and tags it RED / YELLOW / GREEN. Everything is plotted on a shared live map so rescue teams can see where the worst damage is — and where SOS signals are coming from — and launch turn-by-turn navigation to each site.

The app runs in two modes, chosen at launch:

- **Hackathon Demo Mode** — full interactive sandbox with Public, Infrastructure, and Rescue tabs.
- **Active Disaster Event Mode** — a locked command-center view intended for live operations with drone telemetry tied to a specific declared disaster zone.

## Core Features

- **Citizen upload & SOS** — public photo uploads with automatic coordinate lookup from a CSV registry, plus a one-tap emergency SOS beacon that broadcasts a citizen's GPS location to rescue teams.
- **Drone geofence verification** — every upload's coordinates are checked with a Haversine distance calculation against the active drone recon zone; anything outside the radius is auto-rejected.
- **Multi-source infrastructure ingestion** — drone video/photo, satellite imagery, CCTV feed, and bulk dataset-folder ingestion (images + co-located CSVs) via the Infrastructure tab.
- **Local AI damage analysis pipeline** (`backend/image_ai.py`):
  - **YOLOE (prompt-free)** detects infrastructure objects (roads, bridges, buildings, walls, towers, etc.) and separates them from non-infrastructure objects (people, vehicles, animals, vegetation).
  - Detected infrastructure regions are cropped and passed to **CLIP** for zero-shot damage classification across graded prompt sets (intact → minor → moderate → severe → collapsed) plus damage-type prompts (cracks, collapse, deformation, potholes, etc.).
  - A **CLIP scene-level fallback** kicks in when YOLOE finds no infrastructure, so a clearly damaged scene (e.g. a collapsed bridge YOLOE missed) isn't automatically scored as "no damage."
  - Outputs a severity score (0–100%), an infrastructure category, and an annotated image.
- **Video analysis** (`backend/video_ai.py`) — samples frames from uploaded video at a configurable interval and reuses the same image AI engine per frame; the video's overall severity is the worst sampled frame.
- **RED / YELLOW / GREEN triage** — severity ≥70% = RED, ≥40% = YELLOW, else GREEN.
- **Rescue dashboard** — severity breakdown metrics, and SOS signals grouped into geographic hotspots (grid clustering) with a prioritized, navigable response order.
- **Shared geospatial map** (Folium) — Google Roadmap/Satellite layers, the active recon geofence circle, color-coded markers with embedded annotated photos, and one-click Google Maps navigation, shared across all three tabs.
- **Flexible CSV registry** — matches uploaded filenames to CSV coordinate records via exact path, basename, normalized text, or filename-stem matching, tolerant of varying column names (`Image_Path`, `Filename`, `Lat`/`Latitude`, etc.).
- **Persistent shared JSON database** — all uploads, SOS signals, and the active drone zone persist across sessions in a single JSON file.

## Project Structure

```text
Autonomous-Disaster-Response/
├── app.py                      # Entry point / mode router
├── frontend/
│   ├── home.py                 # Mode selection screen
│   ├── sidebar.py               # CSV registry uploads + drone zone info
│   ├── public_tab.py            # Citizen uploads, SOS, public feed
│   ├── infrastructure_tab.py    # Multi-source ingestion + bulk datasets
│   ├── rescue_tab.py             # Severity metrics + SOS hotspot triage
│   └── map_view.py               # Shared Folium map (used by all tabs)
├── backend/
│   ├── config.py                 # DB file / uploads dir config
│   ├── database.py               # Load/save shared JSON DB
│   ├── geospatial.py              # Haversine distance
│   ├── csv_registry.py            # CSV column detection + coordinate lookup
│   ├── media.py                   # Image/video save + base64 encoding
│   ├── ingestion.py                # Upload record creation + geofence check
│   ├── image_ai.py                 # YOLOE + CLIP damage detection pipeline
│   └── video_ai.py                  # Frame sampling + video severity scoring
├── uploaded_media/                  # Saved uploads (gitignored)
├── shared_disaster_db.json          # Persistent shared state
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Model weights

The AI pipeline expects these files at the project root (already present in this repo):

| File | Purpose |
|---|---|
| `yoloe-26s-seg-pf.pt` | **Required.** Prompt-free YOLOE checkpoint used for infrastructure detection. |
| `yoloe-26s-seg.pt` | Standard YOLOE checkpoint (text-prompted) — not used by the current prompt-free pipeline. |
| `yolov8n.pt`, `yolov8s-world.pt` | Auxiliary/legacy YOLO weights. |
| `mobileclip2_b.ts` | Only needed if switching to prompted YOLOE (not required for the prompt-free path). |

`image_ai.py` intentionally uses the **prompt-free** checkpoint (`yoloe-26s-seg-pf.pt`) so it doesn't require MobileCLIP for text prompting — see the comments at the top of that file for why.

> **Note:** these `.pt`/`.ts` weight files are large (tens to hundreds of MB). If you're pushing this to GitHub, consider [Git LFS](https://git-lfs.com/) or excluding them and documenting a download step instead of committing them directly.

CLIP itself (`openai/clip-vit-base-patch32`) is pulled automatically via `transformers` on first run.

GPU (CUDA) is used automatically if available; otherwise the pipeline falls back to CPU.

## CSV Format (Citizen / Infrastructure Registries)

The registry accepts flexible column naming and matches on:

1. Exact filename or full path
2. Basename
3. Normalized text
4. Filename stem (extension-insensitive)

Recognized columns (any of these aliases per field):

- **Image:** `Image_Path`, `Image Path`, `Image`, `Image_Name`, `Filename`, `File_Name`, `File`, `Path`
- **Latitude:** `Latitude`, `Lat`, `GPS_Latitude`
- **Longitude:** `Longitude`, `Lon`, `Long`, `GPS_Longitude`
- **Event** *(optional)*: `Event`, `Disaster`, `Event_Name`

If no match is found, uploads fall back to a default event label and `(0.0, 0.0)`, which will fail geofence verification.

## Data Flow Summary

1. A CSV registry (citizen or infrastructure) is uploaded via the sidebar.
2. A photo/video is uploaded in the Public or Infrastructure tab.
3. `csv_registry.py` resolves coordinates for the uploaded filename.
4. `ingestion.py` checks the coordinates against the active drone geofence (Haversine distance) and records VERIFIED/REJECTED status.
5. Verified images/video frames are run through `image_ai.py` (and `video_ai.py` for video) to produce a severity score, category, and annotated image.
6. Results are tagged RED/YELLOW/GREEN and saved to `shared_disaster_db.json`.
7. The Rescue tab surfaces severity breakdowns and SOS hotspots; `map_view.py` renders everything on a shared live map with navigation links.

## Emergency SOS

From the Public tab, a citizen can send a one-time SOS signal with their coordinates. SOS signals are grouped by rescue teams into geographic hotspots (rounded-coordinate clustering) and ranked by concentration, with a direct Google Maps navigation link to each cluster.
