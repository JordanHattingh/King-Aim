# King Aim v2.6 Full Correctness and Stability Audit

Baseline source snapshot: public `main` / v2.5.2 semantics (`2574428218b2dd7a438cd0f954910ee7678fd961`).

This patch is a correctness/stability freeze. It fixes neural feature contracts, time semantics,
model-bundle compatibility, capture ownership, tracking identity, logging durability and thread
publication. The movement MLP remains a controlled TestArena/accessibility-pointing research
component; this patch does not add or optimize autonomous live enemy-to-mouse/right-stick steering.

## Validation results (Windows — final)

All validation was performed on the clean v2.6 tree after history rewrite.

| Check | Command | Result |
|---|---|---|
| Whitespace | `git diff --check` | PASS |
| Python compile | `python -m py_compile training/*.py training/tests/*.py` | PASS |
| Python contract suite | `python -m unittest discover -s training/tests -p "test_*.py" -v` | 10/10 PASS |
| .NET restore | `dotnet restore ".\Aimmy2.sln"` | PASS |
| Release build | `dotnet build ".\Aimmy2.sln" -c Release --no-restore` | **0 Warning(s) / 0 Error(s)** |
| .NET test suite | `dotnet test ".\Aimmy2.sln" -c Release --no-build` | **59 Passed / 0 Failed / 59 Total** |

Pre-patch baseline for reference: build 0 errors / 172 warnings, tests 36 passed / 3 failed / 39 total.

---

# 1. Three failing C# tests

## 1.1 `CursorExclusion_NullPosition_DoesNotFilterAnything`

**File:** `Aimmy2.Tests/PredictionFilterTests.cs`, around line 123.

### Root cause

The old test created its detection at `(5,5)`. `PredictionFilter` first applies the independent
circular FOV rule centered on `(320,320)` with radius `320`. `(5,5)` is about 445 pixels from the
center, so the detection was correctly rejected before cursor exclusion was evaluated. The test was
not isolating the behavior named by the test.

### Corrected fixture

```csharp
var tensor = MakeSingleDetectionTensor(
    xCenter: 320,
    yCenter: 320,
    width: 10,
    height: 10,
    confidence: 0.9f);
```

Production cursor-null behavior remains unchanged. A null cursor position means cursor exclusion is
not applied.

## 1.2 `SelectedEnemy_DoesNotFlickerBetweenTracks`

**Files:**

- `Aimmy2.Tests/TrackingTests.cs`, around line 110
- `Aimmy2/AILogic/TrackManager.cs`
- `Aimmy2/AILogic/TrackAssociation.cs`

### Root cause

The old pipeline mixed local/model rectangles and screen rectangles, then greedily associated tracks.
Small movement could fail the old match contract and create a replacement `TrackId`. The test helper
also did not fully populate the absolute screen rectangle contract used by the current tracker.

### Corrected design

`Prediction.ScreenRectangle` is now the authoritative absolute desktop-pixel box for tracking. The
tracker uses a timestamped global cost association strategy by default:

```csharp
public ITrackAssociationStrategy AssociationStrategy { get; set; }
    = new GlobalCostTrackAssociation();
```

The test helper now supplies the same absolute rectangle in `ScreenRectangle`. A single smoothly
moving detection remains one track.

## 1.3 `PersistentChallenger_EventuallySwitchesSelection`

**Files:**

- `Aimmy2.Tests/TrackingTests.cs`, around line 131
- `Aimmy2/AILogic/TargetSelector.cs`

### Root cause

The test advanced a synthetic `DateTime now` through `TrackManager`, while the selector's convenience
overload used real `DateTime.UtcNow`. The four test iterations completed in only a few real
milliseconds, so a `40 ms` challenger confirmation timer never elapsed.

### Corrected API

The implicit-wall-clock selector overload was removed. Selection now requires the observation clock:

```csharp
public TargetSelectionResult Select(
    IReadOnlyList<Track> tracks,
    TargetMode mode,
    SemanticRole roleFilter,
    int? fixedTrackId,
    PointF screenCenter,
    float normalizationRadius,
    DateTime now,
    float aimPointFraction = 0.25f)
```

Production and tests now use one time source. A challenger must stay the same best challenger for
`SwitchConfirmationMs` before the switch occurs.

---

# 2. Train/runtime neural contracts

## 2.1 GRU feature encoding

**Runtime:** `Aimmy2/AILogic/TrackRingBuffer.cs`, `FillSequence`, around line 63.

**Training:** `training/train_gru.py`, `_encode`, around line 134.

Both now encode the exact eight features in this order:

```text
0  (cx - 0.5) * 2
1  (cy - 0.5) * 2
2  (log(max(w, 1e-5)) - log_w_mean) / log_w_std
3  (log(max(h, 1e-5)) - log_h_mean) / log_h_std
4  confidence
5  observed_mask
6  (min(dt_seconds, 0.10) - dt_mean) / dt_std
7  (min(age_seconds, 0.25) - age_mean) / age_std
```

The common schema ID is:

```text
track-motion-8x8-v2
```

C# stores it in `NeuralFeatureSchemas.TemporalV2`; Python stores it in
`training/contracts.py::TEMPORAL_FEATURE_SCHEMA`.

## 2.2 `GruNormConstants`

**File:** `Aimmy2/AILogic/ModelManifest.cs`.

The old runtime embedded guessed dataset statistics. Those cannot be guaranteed to match real
training data. This is worse than having no default because a model can load successfully while
receiving the wrong standardized features.

The defaults now fail closed:

```csharp
public float LogWMean { get; set; } = float.NaN;
public float LogWStd  { get; set; } = float.NaN;
// ... all eight values
```

`Validate()` rejects non-finite values and non-positive standard deviations. A temporal model now
requires the exact `norm_constants.json` generated from the **training split only**.

## 2.3 GRU delta output scaling

**Runtime:** `Aimmy2/AILogic/TemporalPredictor.cs`, around line 19.

**Training:** `training/train_gru.py::_make_target`, around lines 154-163.

Training target:

```python
dcx = (next_cx - previous_cx) * 2.0
dcy = (next_cy - previous_cy) * 2.0
```

Runtime conversion is explicitly the mathematical inverse:

```csharp
public const float TrainingDeltaScale = 2f;
public const float RuntimeDeltaInverseScale = 1f / TrainingDeltaScale;

public static float ConvertModelDeltaToScreenFraction(float modelDelta) =>
    modelDelta * RuntimeDeltaInverseScale;
```

The old magic `* 0.5f` is now a named contract and has a C# regression test.

## 2.4 Calibration MLP six-feature contract

**Runtime:** `Aimmy2/AILogic/CalibrationMlp.cs`, `EncodeFeatures`, around line 109.

**Training:** `training/train_calibration.py`, `CalibrationDataset`.

Exact six features:

```text
0 logit(raw_confidence)
1 log(box_area)
2 log(aspect_ratio = h / w)
3 normalized radial distance from center / 0.70710678
4 clamp(frame_age_ms, 0, 500) / 100
5 pose_quality
```

Schema:

```text
detection-context-v2
```

A Python test and C# test now assert the equations.

## 2.5 Movement MLP eight-feature contract

**Runtime:** `Aimmy2/InputLogic/MovementMlp.cs`, `EncodeFeatures`, around line 155.

**Training:** `training/train_movement.py`, `MovementDataset`, around line 44.

Exact order:

```text
0 dx
1 dy
2 distance
3 current_speed_pixels_per_ms
4 target_size_pixels
5 dt_seconds
6 previous_output_vx
7 previous_output_vy
```

Schema:

```text
pointing-velocity-v1
```

`MouseManager` now measures the previously emitted pixel speed and publishes it with `Volatile`, so a
consumer of the controlled pointing model no longer always supplies zero for feature 3.

---

# 3. Timing bugs

## 3.1 Track `DtSeconds`

**File:** `Aimmy2/AILogic/TrackManager.cs`, `UpdateTrackWithDetection`, around line 171.

`dt` is now calculated before `LastSeen` is mutated:

```csharp
PointF oldCenter = TrackCenter(track.BoundingBox);
float dt = Math.Clamp(
    (float)(frameTime - track.LastSeen).TotalSeconds,
    0f,
    0.1f);
```

Only after motion/Kalman calculations does the track publish:

```csharp
track.LastSeen = frameTime;
```

`FirstSeen` is assigned only when a track is created and is not rewritten on observation updates.

## 3.2 `AgeSeconds`

For every real observation:

```csharp
AgeSeconds = 0f
```

For a missing observation, `TrackObservation.Missing` carries forward geometry, sets confidence and
`ObservedMask` to zero, and increments `AgeSeconds` by actual `dt` up to 0.25 seconds. The next real
observation resets age to zero.

This fixes the old semantic bug where track lifetime was accidentally fed as "seconds since last real
detection".

## 3.3 Calibration `FrameAgeMs`

**File:** `Aimmy2/AILogic/AIManager.cs`, around lines 1409-1446.

The capture frame carries timestamps. `FrameAge` is derived as elapsed milliseconds from capture
completion to inference start, and the value passed into `CalibrationSampleInput.FrameAgeMs` remains
milliseconds:

```csharp
float frameAgeMs = (float)FrameAge;
...
FrameAgeMs = frameAgeMs;
```

The calibrator itself divides by `100f` only when encoding feature 4. It does not treat the input as
seconds.

---

# 4. `TrackRingBuffer`

**File:** `Aimmy2/AILogic/TrackRingBuffer.cs`.

Verified/fixed invariants:

- Capacity is exactly `8`.
- `Push` writes at `_head`, increments `_head` modulo eight, and caps `Count` at eight.
- `Tail` uses `(_head - 1 + Capacity) % Capacity`, so it is the newest sample.
- Once full, `_head` points at the oldest sample.
- `GetSequence()`/`FillSequence()` enumerate oldest to newest.
- `FillSequence()` rejects a buffer smaller than 64 floats and rejects a not-ready ring.
- `Reset()` clears the backing array as well as head/count.

Regression test `RingBuffer_WrapsOldestFirst_TailIsNewest_AndCountCapsAtEight` pushes 12 samples and
asserts that the surviving ordered values are samples 4 through 11 and the tail is sample 11.

---

# 5. `AIManager` companion pipeline

## 5.1 `_gruAimHint` stale state

**File:** `Aimmy2/AILogic/AIManager.cs`, around lines 1669-1687.

The hint is cleared **before every new predictor attempt**:

```csharp
_gruAimHint = null;
```

It is set only if the selected track is ready, the model is loaded, norm constants are present and
prediction succeeds. Target loss also clears selection context. A failed GRU call can no longer leave
the previous track's hint alive.

## 5.2 `LoadCompanionModels()` placement and hot swap

**File:** `Aimmy2/AILogic/AIManager.cs`, around lines 613-660.

`_activeManifest` is first resolved from the active model lease/fallback. Only then is
`LoadCompanionModels()` called.

The method now unloads every old companion first:

```csharp
_gruAimHint = null;
_temporalPredictor.Unload();
_calibrationMlp.Unload();
MouseManager.ClearNeuralMovementIf(_movementMlp);
_movementMlp.Unload();
_movementMlp.SetContext(null);
```

This prevents a new vision manifest that omits a companion path from silently retaining the old
bundle's GRU/calibrator/movement session.

## 5.3 Dispose pointer ownership

**Files:**

- `Aimmy2/AILogic/AIManager.cs`, around line 2287
- `Aimmy2/InputLogic/MouseManager.cs`, around line 64

Dispose no longer blindly writes `MouseManager.NeuralMovement = null`. It uses reference identity:

```csharp
MouseManager.ClearNeuralMovementIf(_movementMlp);
```

and:

```csharp
public static bool ClearNeuralMovementIf(MovementMlp expected)
{
    return ReferenceEquals(
        Interlocked.CompareExchange(ref _neuralMovement, null, expected),
        expected);
}
```

If another owner installed a different model, this AIManager cannot clear it.

---

# 6. `TrackLogger`

## 6.1 Raw confidence ordering

**File:** `Aimmy2/AILogic/AIManager.cs`, around lines 1409-1446.

`CalibrationSampleInput.RawConf` is captured from `p.Confidence` before this line:

```csharp
p.Confidence = _calibrationMlp.Calibrate(...);
```

The logged `raw_conf` therefore represents the detector's raw score rather than the calibrated output.

## 6.2 `SyncTracks` and expiry

**File:** `Aimmy2/AILogic/TrackLogger.cs`, around line 128.

`TrackManager.Update()` removes hard-expired tracks before returning the active list. `SyncTracks`
builds `activeIds`; every logger-known track absent from that list is flushed and removed:

```csharp
foreach (int expiredId in _knownTracks.Keys
    .Where(id => !activeIds.Contains(id)).ToList())
{
    FlushTrackSequenceLocked(_knownTracks[expiredId]);
    _knownTracks.Remove(expiredId);
}
```

It appends only a genuinely newer `Tail`, based on `LastBufferTimestamp`, so repeatedly calling the
logger does not duplicate one observation.

## 6.3 GRU JSON fields

**File:** `Aimmy2/AILogic/TrackLogger.cs`, `GruFrame`, around line 427.

Writer field names are exactly:

```text
cx cy w h conf observed dt age
```

They match `training/contracts.py::GRU_FRAME_FIELDS` and `train_gru.py::_encode()`.

## 6.4 Logger thread publication/durability

`LastError` is now published with `Volatile.Read/Write`. Training records are append-only JSONL via one
bounded background writer instead of rewriting a growing JSON array on the inference path. Queue
saturation is counted and exposed as `DroppedWriteBatches`; session stop/flush waits for the writer
control item.

---

# 7. Training pipeline field consistency

A new dependency-free `training/contracts.py` is the Python source for schema IDs and field tuples.
C# mirrors the schema IDs in `NeuralFeatureSchemas.cs`.

## GRU flow

```text
TrackLogger GruFrame
  -> prepare_gru_data.py
  -> session-grouped train/val/test split
  -> train_gru.py
  -> norm_constants.json
  -> trajectory_gru.onnx
  -> update_manifest.py
  -> manifest gru_norm + temporal_feature_schema
```

`prepare_gru_data.py` reads JSONL first, keeps sessions intact and writes per-session sequence files.
`train_gru.py` calculates normalization from the **train split only**. `update_manifest.py` requires
`--gru-norm` whenever `--gru` is provided and validates all eight normalization fields.

## Calibration flow

```text
TrackLogger calibration_samples.jsonl
  -> label_calibration.py
  -> one-to-one detection/GT matching
  -> train_calibration.py
  -> calibration MLP ONNX
  -> update_manifest.py
```

The logger fields expected by the labeling/training path are centralized in
`CALIBRATION_LOG_FIELDS`. Validation is session-grouped by default; random row splitting is allowed
only through an explicit bootstrap flag.

## Movement flow

`record_movement.py` accepts controlled TestArena/accessibility-pointing source data. `train_movement.py`
requires `source == "testarena_pointing"`, uses the exact `MOVEMENT_FEATURE_FIELDS` order and splits by
session.

The Python contract suite now contains 10 passing tests for these paths.

---

# 8. Thread safety

## `MouseManager.NeuralMovement`

**File:** `Aimmy2/InputLogic/MouseManager.cs`, around line 43.

The static reference is now published with `Volatile.Read` and installed with
`Interlocked.Exchange`. `ClearNeuralMovementIf` uses `Interlocked.CompareExchange` for identity-safe
ownership clearing.

## `MouseManager.LastTargetSizePixels`

The static float now uses:

```csharp
get => Volatile.Read(ref _lastTargetSizePixels);
set => Volatile.Write(ref _lastTargetSizePixels, value);
```

The last emitted movement speed is published the same way.

## `AIManager.NeuralStatusText`

**File:** `Aimmy2/AILogic/AIManager.cs`, around line 146.

The backing reference is written/read with `Volatile`:

```csharp
private string _neuralStatusText = "No model loaded";
public string NeuralStatusText => Volatile.Read(ref _neuralStatusText);
```

`UpdateNeuralStatus()` uses `Volatile.Write`. UI timer reads can no longer race with an ordinary
unsynchronized reference publication.

## Companion ONNX sessions

`TemporalPredictor`, `CalibrationMlp` and `MovementMlp` use private locks around session access and
atomic candidate-session swaps. A model cannot be disposed while the same wrapper is inside
`InferenceSession.Run`.

---

# 9. Movement MLP accumulator/context reset

**File:** `Aimmy2/InputLogic/MovementMlp.cs`, around lines 79-92 and 179-190.

New API:

```csharp
public void SetContext(int? contextId)
{
    lock (_sync)
    {
        if (_contextId == contextId)
            return;

        _contextId = contextId;
        ResetMotionStateLocked();
    }
}
```

Context changes reset:

```csharp
_prevVx = 0f;
_prevVy = 0f;
_residualX = 0f;
_residualY = 0f;
```

The reset occurs when:

- task/target identity changes;
- context becomes null on loss;
- a model is loaded;
- a model is unloaded;
- `Reset()` is called;
- the wrapper is disposed.

`AIManager` publishes selected-track context changes and clears the context on loss/model swap.
Residual subpixel state from one context can no longer drift into the next.

---

# Additional stability fixes included

- Pose output validation is decoder/schema aware; a four-keypoint pose output is no longer validated
  as `4 + classCount` detector-only output.
- Keypoint visibility has an explicit manifest flag for logit-vs-activated semantics, preventing
  unconditional double sigmoid.
- Missing class mappings default to `Unknown`, never `Enemy`.
- Track velocity is normalized display units per second, with a time-aware first-order filter.
- Kalman observation timing uses supplied observation timestamps rather than processing wall clock.
- Production capture uses the latest-frame mailbox as the capture/inference handoff; hidden sync
  capture fallback is restricted out of the normal inference path.
- ROI refresh/prediction timing uses elapsed time rather than frame counts/fixed 16 ms assumptions.
- `FrameAge` and capture FPS diagnostics use actual capture timestamps/events.
- Companion manifests validate temporal, calibration and movement feature-schema IDs.
- `Supercluster.KDTree 1.0.4` package reference was removed because the source does not use it and it
  caused NU1701 under .NET 8.
- `Costura.Fody`/`Fody` private package references no longer provide the incomplete `IncludeAssets`
  override that caused the Fody package-reference warning.

# Validation commands

```powershell
dotnet restore ".\Aimmy2.sln"
dotnet build ".\Aimmy2.sln" -c Release --no-restore
dotnet test ".\Aimmy2.sln" -c Release --no-build --logger "console;verbosity=normal"
python -m unittest discover -s training/tests -p "test_*.py" -v
```

All four commands pass on the clean v2.6 tree. See the validation table above for results.
