# Success Metrics and Validation

## Success Metrics
- Performance: 30 FPS or more at 720p on iGPU-only; latency under 120 ms; zero dropped frames over a 30-minute soak test.
- Stability: 6-hour continuous run without crash; auto-recover camera unplug or replug within 5 seconds.
- UX: time-to-first-frame under 2 seconds after app start; filter switch under 200 ms.
- Privacy: no files written; process memory cleared on exit; verified via QA checklist.

## Validation Plan
- Benchmarks: measure FPS and latency with 720p and 1080p on target hardware; profile inference versus compositing time.
- Soak: 6-hour continuous run; monitor stability and temperature.
- Camera unplug or replug test: ensure graceful recovery.
- Visual QA: multiple faces, glasses, hats, varying lighting.
