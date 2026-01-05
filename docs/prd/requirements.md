# Requirements

## Platform and Performance
- Windows 10/11 PC; LG TV at 1080p60 over HDMI; USB3 1080p60 UVC webcam.
- Performance baseline: 720p input at 30–45 FPS acceptable; glass-to-glass latency under 120 ms.
- Stretch: 1080p at 50–60 FPS if a low-cost dGPU is available.

## Reliability and Operations
- Run 6–8 hours continuously without manual restart.
- Auto-recover camera disconnects; watchdog to restart on crash.

## Offline and Privacy
- All models and assets bundled; no network dependency.
- No storage or logging of frames; no uploads.

## Hardware Targets
- Baseline (no dGPU): modern iGPU (Intel Xe or AMD RDNA2 APU), 6C/12T CPU class (e.g., i5-12400 or 5600G), 16 GB RAM, NVMe SSD. Expect about 30–45 FPS at 720p, under 120 ms.
- Upgrade for smoother 1080p60: add low-cost dGPU (GTX 1650, RTX 3050-class, or Intel Arc A380/A750). Expect about 50–60 FPS at 1080p, under 80 ms.
