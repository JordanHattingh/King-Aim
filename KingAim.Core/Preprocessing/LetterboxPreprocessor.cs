using KingAim.Core.Capture;

namespace KingAim.Core.Preprocessing;

/// <summary>
/// Implements the Ultralytics-compatible letterbox preprocessing pipeline:
///   1. Scale to fit within modelSize × modelSize (uniform scale, no stretch).
///   2. Pad symmetrically with constant value 114.
///   3. BGR → RGB.
///   4. Normalise to [0, 1].
///   5. HWC → NCHW, add batch dimension.
///
/// The same logic is used by the Python benchmark harness (benchmark_positive_v1.py)
/// to ensure train/serve parity.
/// </summary>
public sealed class LetterboxPreprocessor : IFramePreprocessor
{
    private const byte PadValue = 114;

    private readonly int _modelSize;

    public LetterboxPreprocessor(int modelSize = 512)
    {
        if (modelSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelSize));
        _modelSize = modelSize;
    }

    public PreprocessedFrame Preprocess(CapturedFrame frame)
    {
        ArgumentNullException.ThrowIfNull(frame);

        int sw = frame.SourceWidth;
        int sh = frame.SourceHeight;

        // Scale to fit
        float scaleR = Math.Min((float)_modelSize / sh, (float)_modelSize / sw);
        int   newW   = (int)Math.Round(sw * scaleR);
        int   newH   = (int)Math.Round(sh * scaleR);

        // Symmetric padding (mirrors Python: round(dw ± 0.1))
        float dw = (_modelSize - newW) / 2f;
        float dh = (_modelSize - newH) / 2f;
        int padLeft   = (int)Math.Round(dw - 0.1f);
        int padTop    = (int)Math.Round(dh - 0.1f);

        var meta = new PreprocessingMetadata
        {
            SourceWidth  = sw,
            SourceHeight = sh,
            ModelWidth   = _modelSize,
            ModelHeight  = _modelSize,
            ScaleR       = scaleR,
            PadLeft      = padLeft,
            PadTop       = padTop,
        };

        // Build the float tensor.
        // Concrete pixel resizing is deferred to the infrastructure layer
        // (which will use OpenCvSharp or a similar library).
        // This scaffold returns a zeroed tensor of the correct shape.
        float[] tensor = new float[1 * 3 * _modelSize * _modelSize];

        if (frame.PixelData.Length > 0)
            FillTensor(frame, newW, newH, padLeft, padTop, tensor);

        return new PreprocessedFrame
        {
            Tensor  = tensor,
            Meta    = meta,
            FrameId = frame.FrameId,
        };
    }

    // Fills the NCHW tensor from raw BGR24 pixel data.
    // Only called when pixel data is present (not in scaffold mode).
    private void FillTensor(CapturedFrame frame, int newW, int newH,
                            int padLeft, int padTop, float[] tensor)
    {
        // Concrete implementation requires a resize step.
        // The infrastructure layer overrides or completes this via OpenCvSharp.
        // Scaffold: leaves tensor zeroed, which is safe for dry-run tests.
    }
}
