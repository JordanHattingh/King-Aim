using System.IO;
using System.Text.Json;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Captures detection and track data to disk for training the GRU, calibration,
    /// and movement neural networks.
    ///
    /// Two output files are written per session (in runs/logs/session_YYYY-MM-DD_HH-mm-ss/):
    ///
    ///   gru_sequences.json
    ///     Array of completed tracks.  Each entry has a "frames" array matching the
    ///     TrackSequenceDataset format expected by training/train_gru.py.
    ///
    ///   calibration_samples.json
    ///     Array of per-detection samples matching training/train_calibration.py.
    ///     The "label" field is written as null — assign 1 (TP) or 0 (FP) in
    ///     post-processing by comparing boxes against ground-truth annotations.
    ///
    /// Usage:
    ///   Enable the "Detection Logging" toggle in the UI.  A session starts
    ///   automatically on the first logged frame and flushes to disk every
    ///   500 completed tracks and on shutdown.
    /// </summary>
    public sealed class TrackLogger : IDisposable
    {
        private string? _outputDir;
        private readonly List<GruSequenceEntry>   _gruSequences      = new();
        private readonly List<CalibrationEntry>    _calibrationSamples = new();
        private readonly Dictionary<int, Track>    _knownTracks       = new();
        private readonly object _lock = new();
        private bool _disposed;
        private int _totalTracksFlushed;
        private int _totalCalibrationSamples;

        public bool IsActive      { get { lock (_lock) { return _outputDir != null; } } }
        public int TotalTracksFlushed        => _totalTracksFlushed;
        public int TotalCalibrationSamples   => _totalCalibrationSamples;
        public string? OutputDir  { get { lock (_lock) { return _outputDir; } } }

        // ── Session control ───────────────────────────────────────────────────

        public void EnsureSessionActive(string logsRootDir)
        {
            lock (_lock)
            {
                if (_outputDir != null) return;
                string timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
                _outputDir = Path.Combine(logsRootDir, $"session_{timestamp}");
                try { Directory.CreateDirectory(_outputDir); }
                catch { _outputDir = null; }
            }
        }

        public void StopSession()
        {
            lock (_lock)
            {
                FlushLockedTracks();
                WriteFilesLocked();
                _outputDir = null;
                _knownTracks.Clear();
                _gruSequences.Clear();
                _calibrationSamples.Clear();
            }
        }

        // ── Per-frame logging ─────────────────────────────────────────────────

        /// <summary>
        /// Log calibration samples for every detection that survived the confidence filter this frame.
        /// Call after PredictionFilter.CreatePredictions() and CalibrationMlp (so raw_conf is the
        /// pre-calibration value — pass rawConf from the Prediction before overwriting it).
        /// </summary>
        public void LogCalibrationSamples(
            IReadOnlyList<CalibrationSampleInput> samples)
        {
            if (!IsActive || samples.Count == 0) return;
            lock (_lock)
            {
                if (_outputDir == null) return;
                foreach (var s in samples)
                {
                    _calibrationSamples.Add(new CalibrationEntry
                    {
                        raw_conf     = Math.Round(s.RawConf,    6),
                        w_norm       = Math.Round(s.WNorm,      6),
                        h_norm       = Math.Round(s.HNorm,      6),
                        cx_norm      = Math.Round(s.CxNorm,     6),
                        cy_norm      = Math.Round(s.CyNorm,     6),
                        frame_age_ms = Math.Round(s.FrameAgeMs, 3),
                        pose_quality = Math.Round(s.PoseQuality,6),
                        label        = null,
                    });
                    _totalCalibrationSamples++;
                }
            }
        }

        /// <summary>
        /// Call after each TrackManager.Update().  Detects tracks that have expired
        /// (disappeared from the active list) and serialises their ring buffer as a GRU sequence.
        /// </summary>
        public void SyncTracks(IReadOnlyList<Track> activeTracks)
        {
            if (!IsActive) return;
            lock (_lock)
            {
                if (_outputDir == null) return;

                var activeIds = new HashSet<int>(activeTracks.Select(t => t.TrackId));

                var expiredIds = _knownTracks.Keys.Where(id => !activeIds.Contains(id)).ToList();
                foreach (int id in expiredIds)
                {
                    FlushTrackSequenceLocked(_knownTracks[id]);
                    _knownTracks.Remove(id);
                }

                foreach (var track in activeTracks)
                    _knownTracks[track.TrackId] = track;

                // Periodic disk flush to prevent data loss on crash
                if (_totalTracksFlushed > 0 && _totalTracksFlushed % 500 == 0)
                    WriteFilesLocked();
            }
        }

        // ── Flush / write ─────────────────────────────────────────────────────

        public void Flush()
        {
            lock (_lock)
            {
                FlushLockedTracks();
                WriteFilesLocked();
            }
        }

        private void FlushLockedTracks()
        {
            foreach (var track in _knownTracks.Values)
                FlushTrackSequenceLocked(track);
            _knownTracks.Clear();
        }

        private void FlushTrackSequenceLocked(Track track)
        {
            var frames = track.RingBuffer.GetSequence().ToList();
            if (frames.Count < 2) return;

            _gruSequences.Add(new GruSequenceEntry
            {
                track_id   = track.TrackId,
                class_name = track.ClassName,
                frames     = frames.Select(f => new GruFrame
                {
                    cx       = Math.Round(f.CxNorm,      8),
                    cy       = Math.Round(f.CyNorm,      8),
                    w        = Math.Round(f.WNorm,       8),
                    h        = Math.Round(f.HNorm,       8),
                    conf     = Math.Round(f.Confidence,  6),
                    observed = (int)f.ObservedMask,
                    dt       = Math.Round(f.DtSeconds,   6),
                    age      = Math.Round(f.AgeSeconds,  6),
                }).ToArray()
            });
            _totalTracksFlushed++;
        }

        private static readonly JsonSerializerOptions _jsonOpts = new()
        {
            WriteIndented     = false,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.Never,
        };

        private void WriteFilesLocked()
        {
            if (_outputDir == null) return;
            try
            {
                File.WriteAllText(
                    Path.Combine(_outputDir, "gru_sequences.json"),
                    JsonSerializer.Serialize(_gruSequences, _jsonOpts));
                File.WriteAllText(
                    Path.Combine(_outputDir, "calibration_samples.json"),
                    JsonSerializer.Serialize(_calibrationSamples, _jsonOpts));
            }
            catch { /* non-fatal — disk full or path invalid */ }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (IsActive) StopSession();
        }

        // ── DTOs ──────────────────────────────────────────────────────────────

        private sealed class GruSequenceEntry
        {
            public int        track_id   { get; set; }
            public string     class_name { get; set; } = "";
            public GruFrame[] frames     { get; set; } = Array.Empty<GruFrame>();
        }

        private sealed class GruFrame
        {
            public double cx       { get; set; }
            public double cy       { get; set; }
            public double w        { get; set; }
            public double h        { get; set; }
            public double conf     { get; set; }
            public int    observed { get; set; }
            public double dt       { get; set; }
            public double age      { get; set; }
        }

        private sealed class CalibrationEntry
        {
            public double  raw_conf     { get; set; }
            public double  w_norm       { get; set; }
            public double  h_norm       { get; set; }
            public double  cx_norm      { get; set; }
            public double  cy_norm      { get; set; }
            public double  frame_age_ms { get; set; }
            public double  pose_quality { get; set; }
            public int?    label        { get; set; }  // null until post-processed
        }
    }

    /// <summary>Pre-calibration detection info for one calibration sample.</summary>
    public readonly struct CalibrationSampleInput
    {
        public float RawConf     { get; init; }
        public float WNorm       { get; init; }
        public float HNorm       { get; init; }
        public float CxNorm      { get; init; }
        public float CyNorm      { get; init; }
        public float FrameAgeMs  { get; init; }
        public float PoseQuality { get; init; }
    }
}
