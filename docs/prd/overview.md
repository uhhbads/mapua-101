# Overview

## Objective
Deliver a zero-cost, fully offline AR booth where students see themselves on an LG TV with overlays: face boxes with random GPA label, dog-ear lens, Y2K low-fi frame, and a custom frame. Keep latency low and the experience smooth with no capture or storage.

## Scope
- In-scope: single USB webcam input; live preview on LG TV over HDMI; face detection with bounding boxes and random GPA text; dog-ear overlay anchored to landmarks; Y2K low-quality frame (downsample plus noise/LUT/scanlines); custom frame overlay; simple UI to switch or auto-rotate filters; offline only; display-only.
- Out-of-scope: cloud services, user accounts, analytics, uploads, printing, multi-camera support.

## Users and UX
- Walk-up students see themselves with filters immediately; under 2 seconds from app start to usable view.
- Minimal UI: on-screen instructions and a filter toggle (keyboard or simple button). No text entry.
- Display-only; no photo or video retention. Provide on-site consent signage stating no data is stored.
