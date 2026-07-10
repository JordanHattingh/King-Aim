namespace Aimmy2.AILogic
{
    /// <summary>
    /// One timestamped observation fed into the GRU.
    /// All positional values are normalized to [0,1] screen fractions before storage.
    /// </summary>
    public readonly struct TrackObservation
    {
        public float CxNorm     { get; init; }   // screen-fraction center x  0..1
        public float CyNorm     { get; init; }   // screen-fraction center y  0..1
        public float WNorm      { get; init; }   // screen-fraction width     0..1
        public float HNorm      { get; init; }   // screen-fraction height    0..1
        public float Confidence { get; init; }   // raw model confidence      0..1
        public float ObservedMask { get; init; } // 1 = real detection, 0 = missing frame
        public float DtSeconds  { get; init; }   // seconds since previous observation
        public float AgeSeconds { get; init; }   // seconds since last REAL detection

        public static TrackObservation Missing(TrackObservation prev, float dt) => new()
        {
            CxNorm      = prev.CxNorm,
            CyNorm      = prev.CyNorm,
            WNorm       = prev.WNorm,
            HNorm       = prev.HNorm,
            Confidence  = 0f,
            ObservedMask = 0f,
            DtSeconds   = dt,
            AgeSeconds  = Math.Clamp(prev.AgeSeconds + dt, 0f, 0.25f),
        };
    }

    /// <summary>
    /// Fixed-capacity 8-frame circular buffer of TrackObservations per track.
    /// The GRU consumes the full sequence [oldest → newest] as a 8×8 float tensor.
    /// </summary>
    public sealed class TrackRingBuffer
    {
        public const int Capacity = 8;
        public const int FeatureCount = 8;

        private readonly TrackObservation[] _buf = new TrackObservation[Capacity];
        private int _head;   // next write position
        private int _count;

        public int Count => _count;
        public bool IsReady => _count >= Capacity;

        public void Push(TrackObservation obs)
        {
            _buf[_head] = obs;
            _head = (_head + 1) % Capacity;
            if (_count < Capacity) _count++;
        }

        /// <summary>
        /// Fills a pre-allocated float array [Capacity × FeatureCount] in oldest-first order.
        /// Caller passes this directly to the GRU ONNX session as a [1, 8, 8] tensor.
        /// </summary>
        public void FillSequence(float[] dest, GruNormConstants norm)
        {
            int oldest = _count < Capacity ? 0 : _head;
            for (int i = 0; i < Capacity; i++)
            {
                var obs = _buf[(oldest + i) % Capacity];
                int base_ = i * FeatureCount;
                dest[base_ + 0] = (obs.CxNorm - 0.5f) * 2f;
                dest[base_ + 1] = (obs.CyNorm - 0.5f) * 2f;
                dest[base_ + 2] = (MathF.Log(MathF.Max(obs.WNorm, 1e-5f)) - norm.LogWMean) / norm.LogWStd;
                dest[base_ + 3] = (MathF.Log(MathF.Max(obs.HNorm, 1e-5f)) - norm.LogHMean) / norm.LogHStd;
                dest[base_ + 4] = obs.Confidence;
                dest[base_ + 5] = obs.ObservedMask;
                dest[base_ + 6] = (Math.Clamp(obs.DtSeconds,  0f, 0.10f) - norm.DtMean)  / norm.DtStd;
                dest[base_ + 7] = (Math.Clamp(obs.AgeSeconds, 0f, 0.25f) - norm.AgeMean) / norm.AgeStd;
            }
        }

        public void Reset()
        {
            _head = 0;
            _count = 0;
        }
    }
}
