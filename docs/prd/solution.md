# Solution Approach

## Models (Zero Cost)
- MediaPipe Face Detection (BlazeFace) for bounding boxes.
- MediaPipe Face Mesh for 468 landmarks (dog ears and nose anchoring, head pose). Both on-device.

## Rendering Pipeline
- Camera capture.
- GPU convert NV12/YUV.
- Inference at 640–720p.
- Overlays in a single GPU pass: bounding boxes with GPA text, dog-ear sprites anchored to landmarks, Y2K or custom frame, post-LUT plus downsample or upsample plus noise or scanlines.
- Output to swapchain and LG TV.

## Tech Stack Options (Free)
- Native: C++ with DirectX 11/12 or Vulkan; MediaPipe C++ GPU delegate.
- Desktop web: Electron or Chromium with WebGPU/WebGL plus MediaPipe JS; GPU-accelerated shaders for filters.
- Pick based on team strength.

## Assets
- PNG or SVG dog ears and nose with alpha.
- Y2K frame PNG plus LUT.
- Custom frame PNG.
- Font for GPA text.
