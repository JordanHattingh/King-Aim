# Current Vision Architecture Audit

This document audits the vision/aim pipeline in Aimmy2 as it exists prior to the
accessibility-focused gamepad-assist work. It covers capture, inference, target
selection, thread ownership, and output.

## 1. Thread ownership

`AIManager` owns a single long-running `Task` (`_aiLoopTask`, created with
`TaskCreationOptions.LongRunning`) started from `LoadModelAsync` after a model
successfully loads. This task runs `AiLoopAsync`, which loops until
`_isAiLoopRunning` is cleared or the `CancellationTokenSource` is cancelled.

There is no separate capture thread, inference thread, or output thread — capture,
inference, target selection, overlay updates, auto-trigger, and mouse movement all
run inline, sequentially, within the same loop iteration on the same task. The only
concurrency primitive is `_inferenceGate` (a `SemaphoreSlim(1,1)`), which exists to
let `RunPerformanceBenchmarkAsync` safely borrow the same inference path from a
different caller without racing the main loop — it is not used to parallelize
capture and inference.

A model reload/hot-swap does not currently exist: `LoadModelAsync` starts the loop
task once; there is no atomic "load candidate, validate, swap, dispose old" flow.

## 2. Capture flow

`CaptureManager` performs synchronous screen capture via two backends:

- **DirectX (DXGI Desktop Duplication)** — `InitializeDxgiDuplication()` sets up a
  D3D11 device and `IDXGIOutputDuplication` for the current display. `DirectX(...)`
  acquires the next frame (`AcquireNextFrame`), copies the relevant sub-region into
  a staging texture, maps it, and blits into a reusable `Bitmap` (`directXBitmap`),
  then returns a **clone** of that bitmap. A small time-boxed cache
  (`_cachedFrame`, 15ms timeout) is used to serve a very recent frame when
  `AcquireNextFrame` times out (i.e., no new frame yet) or on transient device
  errors.
- **GDI+** — `GDIScreen(...)` uses `Graphics.CopyFromScreen` into a reusable
  `Bitmap`, also returning a clone.

`AIManager.GetClosestPrediction()` calls `_captureManager.ScreenGrab(detectionBox, ...)`
**synchronously, inline, inside the AI loop iteration**, blocking the loop until a
frame (or null) comes back. There is no frame queue, no mailbox, and no
producer/consumer separation between capture and inference — capture latency
directly extends every loop iteration.

The returned `Bitmap` is caller-owned; `GetClosestPrediction` disposes it in a
`finally` block after inference and (optionally) `SaveFrame` complete.

## 3. Inference flow

Still inline within `GetClosestPrediction()`:

1. The captured `Bitmap` is converted into a reusable `float[]` via
   `BitmapToFloatArrayInPlace` (in `MathUtil`), sized `3 * IMAGE_SIZE * IMAGE_SIZE`.
2. The float array is copied into a reusable `DenseTensor<float>` (recreated only
   if image size changes, otherwise the buffer is copied in-place to avoid
   reallocation).
3. `_onnxModel.Run(_reusableInputs, _outputNames, _modeloptions)` executes
   synchronously on the calling (AI loop) thread/task. `OnnxModelSessionFactory`
   creates the `InferenceSession` with either the DirectML or CPU execution
   provider (DirectML attempted first, CPU as fallback) — this choice is made once
   at model load time in `AIManager.InitializeModel`.
4. The output tensor is handed to `PredictionFilter.CreatePredictions(...)`, which
   decodes the YOLOv8-style output layout (`[1, 4+numClasses, numDetections]`) into
   a `List<Prediction>`, applying confidence thresholding and FOV-box filtering
   (`fovMinX/MaxX/MinY/MaxY`, computed from `AimSettings.FovSize`) as it goes. There
   is no batching and no separate inference worker/thread — this all happens on
   the AI loop's task.

Model metadata (`names` custom metadata field, parsed as JSON) is used to populate
`_modelClasses`; there is no manifest file, no semantic role concept, and models
without a `names` field only get a single implicit class ("enemy" from the
constructor's default `_modelClasses` dictionary, effectively kept if metadata
loading fails).

## 4. Target selection flow

After `PredictionFilter` produces `List<Prediction> KDPredictions`:

1. A simple linear scan (comment notes a KD-tree was previously used and was
   replaced) finds the prediction whose center is closest to the image center —
   `bestCandidate`.
2. `bestCandidate` and the full `KDPredictions` list are passed into
   `StickyAimSelector.SelectTarget(...)`, which implements a single-target sticky
   lock: it looks for the detection nearest the crosshair, and if a target is
   already locked, decides whether the new nearest-to-crosshair detection is
   "the same" target by checking distance-to-last-position against a
   size-scaled tracking radius (`targetSize * 3`) and a bounding-box area ratio
   (`sizeRatio > 0.5`). If they match, velocity is smoothed and the lock persists;
   if not, a short grace period (`_framesWithoutMatch >= 3` or very-centered
   override) is used before switching to a new lock. When there are no
   detections at all for up to `MaxFramesWithoutTarget` (3) frames, it
   extrapolates the last known position using smoothed velocity before giving up
   and resetting.
3. There is only ever one tracked target at a time — no `List<Track>`, no stable
   numeric track IDs across the app, no semantic roles (every detected class is
   implicitly treated the same way; `AimSettings.TargetClass` just filters which
   class's confidence channel is scored during decode, it doesn't distinguish
   "enemy" from "player"/"friendly" downstream). "Best Confidence" vs a specific
   class name is the only selection axis exposed to the user via
   `AimSettings.TargetClass`.

## 5. Output flow

Once `StickyAimSelector` returns a final `Prediction`:

- `CalculateCoordinates(...)` converts the prediction's box into absolute screen
  coordinates (`detectedX`/`detectedY`), applying X/Y offsets (fixed pixel or
  percentage-based) and the configured aiming boundary alignment (Top/Center/
  Bottom), and — if `ShowDetectedPlayer` is on — updates the on-screen overlay
  (confidence label, tracer line, focus box) via `Application.Current.Dispatcher`.
- `AutoTrigger()` (before coordinate calculation, inline in `AiLoopAsync`) fires a
  synthetic click either unconditionally, only when the mouse cursor is already
  inside the last detection box (`CursorCheck`), or continuously while a target is
  present (`SprayMode`), gated on the aim keybinds being held / constant tracking
  being enabled.
- `HandleAim(...)` is the only path that ever moves the mouse. If aim assist is
  active (keybind held, or constant tracking), it either calls
  `MouseManager.MoveCrosshair(detectedX, detectedY)` directly, or — if
  `AimSettings.Predictions` is enabled — routes the raw detection through one of
  three selectable predictors (Kalman filter, "Shall0e's Prediction", or
  "wisethef0x's EMA Prediction") to get a lead-compensated position before calling
  `MouseManager.MoveCrosshair`.

There is currently **no gamepad output of any kind** — the only actuator is
`MouseManager` driving simulated mouse movement/clicks. There is no
`IGamepadOutput` abstraction, no ViGEm integration, and no separate "vision result
→ controller stick" translation layer; mouse movement is computed and applied in
the same inline step as overlay/tracer updates.

## 6. Summary of gaps this work addresses

| Gap | Current state |
|---|---|
| Capture/inference coupling | Synchronous, inline, single task — no mailbox |
| Track identity | None — single implicit sticky lock, no stable IDs |
| Semantic roles | None — classes are just confidence channels/names |
| Target mode | Single implicit mode (`TargetClass` string) |
| Output actuator | Mouse only, no gamepad path |
| Model packaging | Class names read from ONNX metadata only, no manifest/versioning |
| Hot-swap | Not supported; model is loaded once at construction |
