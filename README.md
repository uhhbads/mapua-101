# MAPUA 101

## Features
- **GPA Scanner** - Face detection with bounding boxes and random GPA overlay (persistent per person)
- **Dog Filter** - Snapchat-style dog ears and nose anchored to face landmarks
- **Y2K Retro** - Low-fi CRT aesthetic with pixelation, scanlines, noise, and vintage color grading
- **Custom Frame** - Overlay your own PNG frame with transparency (to add)

## Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the Application
```powershell
python src/main.py
```

### 3. Controls
| Key | Action |
|-----|--------|
| **SPACE** | Next filter |
| **1-4** | Select filter directly |
| **A** | Toggle auto-rotate |
| **H** | Show/hide instructions |
| **F** | Toggle fullscreen |
| **Q / ESC** | Quit |

## Configuration

Edit `config.json` to adjust settings:

```json
{
    "camera_index": 0,
    "capture_width": 1280,
    "capture_height": 720,
    "display_fullscreen": true,
    "target_fps": 30,
    "filter_auto_rotate": true,
    "filter_rotate_interval_seconds": 15,
    "gpa_refresh_interval_seconds": 3,
    "show_fps": true,
    "show_instructions": true
}
```

| Setting | Description |
|---------|-------------|
| `camera_index` | Camera device index (default: 0) |
| `capture_width/height` | Resolution (default: 1280x720) |
| `display_fullscreen` | Fullscreen mode (default: true) |
| `filter_auto_rotate` | Auto-cycle filters (default: true) |
| `filter_rotate_interval_seconds` | Rotation interval (default: 15s) |
| `show_fps` | Display FPS counter (default: true) |

## Custom Assets

### Custom Frame
Replace `assets/frames/custom_frame.png` with your own PNG:
- Must have alpha channel (transparency) for center
- Will be auto-scaled to display resolution

### Dog Filter Assets
Replace in `assets/overlays/`:
- `dog_ear_left.png` - Left ear (RGBA)
- `dog_ear_right.png` - Right ear (RGBA)  
- `dog_nose.png` - Nose (RGBA)

## Hardware Requirements
- Windows 10/11 PC
- USB webcam (1080p60 recommended)
- Display/TV connected via HDMI
- Modern iGPU or dGPU for smooth performance

## Project Structure
```
mapua_101/
├── config.json              # Runtime configuration
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── PLAN.md                  # PRD index
├── TASKS.md                 # Implementation tasks
├── docs/prd/                # PRD documents
├── src/                     # Source code
│   ├── main.py              # Entry point
│   ├── config.py            # Configuration loader
│   ├── camera.py            # Camera capture
│   ├── display.py           # Display handler with overlays
│   └── filters/             # Filter modules
│       ├── base.py          # Base filter + manager
│       ├── face_detection.py # MediaPipe face detector
│       ├── face_mesh.py     # MediaPipe face landmarks
│       ├── gpa_filter.py    # GPA Scanner filter
│       ├── dog_filter.py    # Dog ears/nose filter
│       ├── y2k_filter.py    # Y2K retro filter
│       └── custom_frame_filter.py # Custom frame overlay
└── assets/
    ├── models/              # MediaPipe models (auto-downloaded)
    ├── frames/              # Frame overlays
    └── overlays/            # Filter sprites
```

## Booth Setup
1. Connect webcam via USB3
2. Connect display/TV via HDMI
3. Run `python src/main.py`
4. Position camera facing the viewing area
5. Let students enjoy!

## Privacy
- **Display-only** - No photos or videos are saved
- **Offline** - No network connection required
- **No data storage** - Frames are processed in memory only
