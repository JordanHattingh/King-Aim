# Accessibility Outputs

All output policies consume `AccessibilityObservation`: stable track ID, role, desktop and normalized centers, confidence, velocity, GRU-predicted center, keypoints, occlusion, extrapolation, and age.

Visual output owns contrast, color, markers, arrows, confidence, predicted styling, reduced motion, and scaling. Haptic output owns direction, intensity, pulse frequency, acquisition, and occlusion patterns. Audio output owns stereo pan, pitch, pulse rate, acquisition tones, speech-free mode, and volume.

Controlled pointing is optional and separately enabled. It enforces strength, speed, acceleration, dead-zone, observation-age, semantic-role, manual-override, emergency-disable, and fail-closed model gates.
