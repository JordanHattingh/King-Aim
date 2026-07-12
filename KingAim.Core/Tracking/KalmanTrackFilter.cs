using KingAim.Core.Perception;

namespace KingAim.Core.Tracking;

/// <summary>
/// Per-track Kalman filter with state [cx, cy, w, h, vx, vy].
/// Constant-velocity model with white-noise-acceleration process noise.
/// Velocity (vx, vy) is in pixels per second.
/// </summary>
internal sealed class KalmanTrackFilter
{
    // State vector [cx, cy, w, h, vx, vy]
    private readonly float[] _x = new float[6];
    // Covariance matrix 6×6 row-major
    private readonly float[] _P = new float[36];
    // Scratch buffer used during Predict to avoid allocation
    private readonly float[] _T = new float[36];

    /// <summary>Process noise: acceleration standard deviation in pixels/s².</summary>
    public float AccelerationNoise { get; set; } = 500f;
    /// <summary>Measurement noise: bounding-box coordinate standard deviation in pixels.</summary>
    public float MeasurementNoise  { get; set; } = 10f;
    /// <summary>Independent size noise added each frame in pixels.</summary>
    public float SizeNoise         { get; set; } = 2f;

    public float Cx => _x[0];
    public float Cy => _x[1];
    public float W  => MathF.Max(_x[2], 1f);
    public float H  => MathF.Max(_x[3], 1f);
    public float Vx => _x[4];
    public float Vy => _x[5];

    public void Initialize(DetectionBoundingBox box)
    {
        _x[0] = box.CentreX; _x[1] = box.CentreY;
        _x[2] = box.Width;   _x[3] = box.Height;
        _x[4] = 0f;          _x[5] = 0f;

        Array.Clear(_P, 0, 36);
        _P[0]  = 100f;      // cx variance (10 px sigma)
        _P[7]  = 100f;      // cy
        _P[14] = 100f;      // w
        _P[21] = 100f;      // h
        _P[28] = 250_000f;  // vx (very uncertain at init: ~500 px/s sigma)
        _P[35] = 250_000f;  // vy
    }

    /// <summary>
    /// Predict state forward by <paramref name="dtSeconds"/>.
    /// Returns the predicted bounding box for IoU association.
    /// </summary>
    public DetectionBoundingBox Predict(double dtSeconds)
    {
        float dt  = (float)Math.Clamp(dtSeconds, 1.0 / 240.0, 0.5);
        float dt2 = dt * dt;
        float dt3 = dt2 * dt;
        float dt4 = dt3 * dt;

        // x = F·x  (positions advance by velocity×dt; size and velocity unchanged)
        _x[0] += _x[4] * dt;
        _x[1] += _x[5] * dt;

        // P = F·P·Fᵀ + Q
        // F has non-trivial off-diagonals only for rows/cols 0↔4 and 1↔5.
        //
        // Step 1: T = F·P  — only rows 0 and 1 change
        //   T[0,j] = P[0,j] + dt·P[4,j]
        //   T[1,j] = P[1,j] + dt·P[5,j]
        Array.Copy(_P, _T, 36);
        for (int j = 0; j < 6; j++)
        {
            _T[j]     = _P[j]      + dt * _P[24 + j]; // row 0
            _T[6 + j] = _P[6 + j]  + dt * _P[30 + j]; // row 1
        }

        // Step 2: P = T·Fᵀ  — only cols 0 and 1 change
        //   P[i,0] = T[i,0] + dt·T[i,4]
        //   P[i,1] = T[i,1] + dt·T[i,5]
        for (int i = 0; i < 6; i++)
        {
            _P[i * 6 + 0] = _T[i * 6 + 0] + dt * _T[i * 6 + 4];
            _P[i * 6 + 1] = _T[i * 6 + 1] + dt * _T[i * 6 + 5];
            _P[i * 6 + 2] = _T[i * 6 + 2];
            _P[i * 6 + 3] = _T[i * 6 + 3];
            _P[i * 6 + 4] = _T[i * 6 + 4];
            _P[i * 6 + 5] = _T[i * 6 + 5];
        }

        // Step 3: P += Q  (CWNA: cx/vx and cy/vy pairs; independent size noise)
        float a2 = AccelerationNoise * AccelerationNoise;
        float s2 = SizeNoise * SizeNoise;
        _P[0 * 6 + 0] += a2 * dt4 / 4f;  // cx–cx
        _P[0 * 6 + 4] += a2 * dt3 / 2f;  // cx–vx
        _P[4 * 6 + 0] += a2 * dt3 / 2f;  // vx–cx
        _P[4 * 6 + 4] += a2 * dt2;        // vx–vx
        _P[1 * 6 + 1] += a2 * dt4 / 4f;  // cy–cy
        _P[1 * 6 + 5] += a2 * dt3 / 2f;  // cy–vy
        _P[5 * 6 + 1] += a2 * dt3 / 2f;  // vy–cy
        _P[5 * 6 + 5] += a2 * dt2;        // vy–vy
        _P[2 * 6 + 2] += s2;              // w–w
        _P[3 * 6 + 3] += s2;              // h–h

        float halfW = MathF.Max(_x[2], 1f) / 2f;
        float halfH = MathF.Max(_x[3], 1f) / 2f;
        return new DetectionBoundingBox(
            _x[0] - halfW, _x[1] - halfH,
            _x[0] + halfW, _x[1] + halfH);
    }

    /// <summary>Correct state using the matched detection's bounding box.</summary>
    public void Correct(DetectionBoundingBox det)
    {
        // Innovation y = z − H·x  where H = [I₄ | 0₄ₓ₂]
        float y0 = det.CentreX - _x[0];
        float y1 = det.CentreY - _x[1];
        float y2 = det.Width   - _x[2];
        float y3 = det.Height  - _x[3];

        // S = H·P·Hᵀ + R  →  upper-left 4×4 of P + R·I₄
        float r2 = MeasurementNoise * MeasurementNoise;
        Span<float> S    = stackalloc float[16];
        Span<float> Sinv = stackalloc float[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                S[i * 4 + j] = _P[i * 6 + j];
        S[0]  += r2;
        S[5]  += r2;
        S[10] += r2;
        S[15] += r2;

        if (!Inv4x4(S, Sinv)) return; // singular S — skip correction

        // K = P·Hᵀ·S⁻¹  →  first-4-columns of P multiplied by S⁻¹  (6×4 result)
        Span<float> K = stackalloc float[24];
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 4; j++)
            {
                float s = 0f;
                for (int k = 0; k < 4; k++)
                    s += _P[i * 6 + k] * Sinv[k * 4 + j];
                K[i * 4 + j] = s;
            }

        // x += K·y
        for (int i = 0; i < 6; i++)
            _x[i] += K[i * 4] * y0 + K[i * 4 + 1] * y1
                   + K[i * 4 + 2] * y2 + K[i * 4 + 3] * y3;

        // P = (I − K·H)·P
        // (I − K·H)[i,k] = δᵢₖ − (k < 4 ? K[i,k] : 0)
        // New P[i,j] = P[i,j] − ΣₖK[i,k]·P[k,j]  for k in 0..3
        Span<float> Pnew = stackalloc float[36];
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
            {
                float s = _P[i * 6 + j];
                for (int k = 0; k < 4; k++)
                    s -= K[i * 4 + k] * _P[k * 6 + j];
                Pnew[i * 6 + j] = s;
            }
        Pnew.CopyTo(_P);

        // Enforce symmetry and positive diagonal (numerical stability)
        for (int i = 0; i < 6; i++)
        {
            if (_P[i * 6 + i] < 0.01f) _P[i * 6 + i] = 0.01f;
            for (int j = i + 1; j < 6; j++)
            {
                float avg = (_P[i * 6 + j] + _P[j * 6 + i]) * 0.5f;
                _P[i * 6 + j] = avg;
                _P[j * 6 + i] = avg;
            }
        }

        // Guard against NaN/Inf propagation
        for (int n = 0; n < 6; n++)
            if (!float.IsFinite(_x[n])) _x[n] = 0f;
        if (_x[2] < 1f) _x[2] = 1f;
        if (_x[3] < 1f) _x[3] = 1f;
    }

    /// <summary>Cap diagonal variances to prevent unbounded growth during missed frames.</summary>
    public void CapCovariance(float maxDiag = 1_000_000f)
    {
        for (int i = 0; i < 6; i++)
            if (_P[i * 6 + i] > maxDiag) _P[i * 6 + i] = maxDiag;
    }

    private static bool Inv4x4(Span<float> m, Span<float> inv)
    {
        Span<float> a = stackalloc float[16];
        m.CopyTo(a);
        inv.Clear();
        inv[0] = inv[5] = inv[10] = inv[15] = 1f;

        for (int col = 0; col < 4; col++)
        {
            // Partial pivot
            int   pivot  = col;
            float maxVal = MathF.Abs(a[col * 4 + col]);
            for (int row = col + 1; row < 4; row++)
            {
                float v = MathF.Abs(a[row * 4 + col]);
                if (v > maxVal) { maxVal = v; pivot = row; }
            }
            if (maxVal < 1e-8f) return false;

            if (pivot != col)
            {
                for (int j = 0; j < 4; j++)
                {
                    (a[col * 4 + j],   a[pivot * 4 + j])   = (a[pivot * 4 + j],   a[col * 4 + j]);
                    (inv[col * 4 + j], inv[pivot * 4 + j]) = (inv[pivot * 4 + j], inv[col * 4 + j]);
                }
            }

            float diag = a[col * 4 + col];
            for (int j = 0; j < 4; j++) { a[col * 4 + j] /= diag; inv[col * 4 + j] /= diag; }

            for (int row = 0; row < 4; row++)
            {
                if (row == col) continue;
                float fac = a[row * 4 + col];
                for (int j = 0; j < 4; j++)
                {
                    a[row * 4 + j]   -= fac * a[col * 4 + j];
                    inv[row * 4 + j] -= fac * inv[col * 4 + j];
                }
            }
        }
        return true;
    }
}
