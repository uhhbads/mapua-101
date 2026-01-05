# Implementation Tasks

## Batch 1: Project Setup and Basic Camera Feed
- [x] Initialize Python project structure
- [x] Create requirements.txt (opencv, mediapipe, numpy)
- [x] Create main entry point with camera capture
- [x] Display live feed in fullscreen window
- [x] Add basic FPS counter

## Batch 2: Face Detection with GPA Overlay
- [x] Integrate MediaPipe Face Detection
- [x] Draw bounding boxes around detected faces
- [x] Generate and display random GPA per face
- [x] Style text with legible font/contrast

## Batch 3: Dog-Ear Lens Filter
- [x] Integrate MediaPipe Face Mesh for landmarks
- [x] Load dog ear/nose PNG assets
- [x] Anchor sprites to landmark positions
- [x] Add smoothing for natural movement

## Batch 4: Y2K Low-Fi Frame Effect
- [x] Create Y2K border frame asset
- [x] Implement downsample/upsample pipeline
- [x] Add noise, scanlines, chromatic aberration
- [x] Apply color LUT for vintage tint

## Batch 5: Custom Frame Overlay
- [x] Load custom frame PNG with transparency
- [x] Composite frame over video feed
- [x] Ensure proper scaling to display resolution

## Batch 6: Filter Switching and Polish
- [x] Add keyboard controls for filter cycling
- [x] Implement auto-rotate timer
- [x] Add on-screen instructions
- [x] Create configuration file for settings
- [x] Final performance tuning

## Stack Choice
Using Python + OpenCV + MediaPipe:
- Zero cost, open source
- Fast prototyping
- MediaPipe GPU acceleration available
- Cross-platform, easy to run on any Windows PC
