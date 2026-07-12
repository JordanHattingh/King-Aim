# Architecture

King Aim separates perception, temporal state, semantic observations, and output policy. `AIManager` decodes detector or pose tensors, calibrates raw confidence, and hands accepted detections to `TrackManager`. Global Hungarian association combines IoU, center distance, timestamped Kalman motion, optional GRU motion, and optional pose geometry. Tracks publish immutable `AccessibilityObservation` values; output modules never mutate tracking state.

The temporal contract is eight frames by eight features: normalized center, normalized size, confidence, observed mask, delta time, and age since the last real observation. Companion schemas are `track-motion-8x8-v2`, `detection-context-v2`, and `pointing-velocity-v1`. Incompatible schemas fail closed.

The candidate pose contract is one `enemy` class with `head`, `neck`, `upper_chest`, and `hip` in that order.
