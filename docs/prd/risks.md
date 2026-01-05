# Risks and Mitigations

- iGPU underperforms at 1080p. Mitigation: cap inference to 640–720p, upscale the composite; optional cheap dGPU.
- Asset quality or poor anchoring. Mitigation: test landmarks for multiple head poses; tweak anchor offsets and scales; add light smoothing.
- Latency spikes from CPU to GPU copies. Mitigation: keep the pipeline GPU-resident; avoid CPU readbacks; disable vsync if necessary.
