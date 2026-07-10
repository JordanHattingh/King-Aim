using System.IO;
using Other;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using LogLevel = Other.LogManager.LogLevel;

namespace Aimmy2.AILogic
{
    /// <summary>
    /// Collects temporal and confidence-calibration data without changing inference decisions.
    /// Training records are serialized as append-only JSONL by one background writer so the
    /// inference path never rewrites a growing JSON array or waits on disk I/O.
    /// </summary>
    public sealed class TrackLogger : IDisposable
    {
        private const int MaxFramesPerLoggedSequence = 4096;
        private const int TemporalOverlapFrames = TrackRingBuffer.Capacity;
        private const int WriterCapacity = 8192;

        private readonly Dictionary<int, LoggedTrackState> _knownTracks = new();
        private readonly object _lock = new();
        private readonly Channel<LogWriteItem> _writerChannel;
        private readonly CancellationTokenSource _writerCancellation = new();
        private readonly Task _writerTask;

        private string? _outputDir;
        private bool _disposed;
        private int _totalTracksFlushed;
        private int _totalCalibrationSamples;
        private long _droppedWriteBatches;

        public TrackLogger()
        {
            _writerChannel = Channel.CreateBounded<LogWriteItem>(new BoundedChannelOptions(WriterCapacity)
            {
                SingleReader = true,
                SingleWriter = false,
                FullMode = BoundedChannelFullMode.Wait,
            });
            _writerTask = Task.Run(() => WriterLoopAsync(_writerCancellation.Token));
        }

        public bool IsActive { get { lock (_lock) { return _outputDir != null; } } }
        public int TotalTracksFlushed => Volatile.Read(ref _totalTracksFlushed);
        public int TotalCalibrationSamples => Volatile.Read(ref _totalCalibrationSamples);
        public long DroppedWriteBatches => Interlocked.Read(ref _droppedWriteBatches);
        public string? OutputDir { get { lock (_lock) { return _outputDir; } } }
        private string? _lastError;
        public string? LastError => Volatile.Read(ref _lastError);

        private void SetLastError(string? value) => Volatile.Write(ref _lastError, value);

        public void EnsureSessionActive(string logsRootDir)
        {
            lock (_lock)
            {
                ThrowIfDisposed();
                if (_outputDir != null)
                    return;

                string timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd_HH-mm-ss-fff");
                _outputDir = Path.Combine(logsRootDir, $"session_{timestamp}");
                try
                {
                    Directory.CreateDirectory(_outputDir);
                    SetLastError(null);
                    WriteSessionMetadataLocked();
                }
                catch (Exception ex)
                {
                    SetLastError(ex.Message);
                    _outputDir = null;
                    LogManager.Log(LogLevel.Warning, $"Training logger could not start: {ex.Message}");
                }
            }
        }

        public void StopSession()
        {
            TaskCompletionSource<bool>? flushCompletion = null;
            lock (_lock)
            {
                if (_outputDir == null)
                    return;

                FlushLockedTracks();
                flushCompletion = EnqueueFlushLocked(_outputDir);
                _outputDir = null;
                _knownTracks.Clear();
            }

            WaitForFlush(flushCompletion);
        }

        public void LogCalibrationSamples(IReadOnlyList<CalibrationSampleInput> samples)
        {
            if (samples.Count == 0)
                return;

            lock (_lock)
            {
                if (_outputDir == null || _disposed)
                    return;

                string payload = string.Join('\n', samples.Select(s => JsonSerializer.Serialize(new CalibrationEntry
                {
                    frame_id = s.FrameId,
                    detection_index = s.DetectionIndex,
                    raw_conf = Math.Round(s.RawConf, 6),
                    w_norm = Math.Round(s.WNorm, 6),
                    h_norm = Math.Round(s.HNorm, 6),
                    cx_norm = Math.Round(s.CxNorm, 6),
                    cy_norm = Math.Round(s.CyNorm, 6),
                    frame_age_ms = Math.Round(s.FrameAgeMs, 3),
                    pose_quality = Math.Round(s.PoseQuality, 6),
                    label = null,
                }, JsonOptions)));

                if (EnqueueAppendLocked(Path.Combine(_outputDir, "calibration_samples.jsonl"), payload))
                    Interlocked.Add(ref _totalCalibrationSamples, samples.Count);
            }
        }

        /// <summary>
        /// Appends at most one newly-published TrackObservation per active track and flushes tracks
        /// that have expired. The logger keeps its own bounded long-form history for GRU training.
        /// </summary>
        public void SyncTracks(IReadOnlyList<Track> activeTracks)
        {
            lock (_lock)
            {
                if (_outputDir == null || _disposed)
                    return;

                var activeIds = new HashSet<int>(activeTracks.Select(t => t.TrackId));
                foreach (int expiredId in _knownTracks.Keys.Where(id => !activeIds.Contains(id)).ToList())
                {
                    FlushTrackSequenceLocked(_knownTracks[expiredId]);
                    _knownTracks.Remove(expiredId);
                }

                foreach (Track track in activeTracks)
                {
                    if (!_knownTracks.TryGetValue(track.TrackId, out LoggedTrackState? state))
                    {
                        state = new LoggedTrackState(track.TrackId, track.ClassName);
                        _knownTracks.Add(track.TrackId, state);
                    }

                    state.ClassName = track.ClassName;
                    if (track.RingBuffer.Count > 0 && track.LastBufferTimestamp > state.LastObservationTimestamp)
                    {
                        state.Frames.Add(track.RingBuffer.Tail);
                        state.LastObservationTimestamp = track.LastBufferTimestamp;
                    }

                    if (state.Frames.Count >= MaxFramesPerLoggedSequence)
                        FlushChunkAndKeepOverlapLocked(state);
                }
            }
        }

        public void Flush()
        {
            TaskCompletionSource<bool>? flushCompletion;
            lock (_lock)
            {
                if (_outputDir == null || _disposed)
                    return;

                FlushLockedTracks();
                flushCompletion = EnqueueFlushLocked(_outputDir);
            }

            WaitForFlush(flushCompletion);
        }

        private void FlushLockedTracks()
        {
            foreach (LoggedTrackState state in _knownTracks.Values)
                FlushTrackSequenceLocked(state);
            _knownTracks.Clear();
        }

        private void FlushChunkAndKeepOverlapLocked(LoggedTrackState state)
        {
            TrackObservation[] overlap = state.Frames
                .TakeLast(Math.Min(TemporalOverlapFrames, state.Frames.Count))
                .ToArray();
            FlushTrackSequenceLocked(state);
            state.Frames.Clear();
            state.Frames.AddRange(overlap);
        }

        private void FlushTrackSequenceLocked(LoggedTrackState state)
        {
            if (_outputDir == null || state.Frames.Count < TrackRingBuffer.Capacity + 1)
                return;

            var entry = new GruSequenceEntry
            {
                track_id = state.TrackId,
                class_name = state.ClassName,
                frames = state.Frames.Select(f => new GruFrame
                {
                    cx = Math.Round(f.CxNorm, 8),
                    cy = Math.Round(f.CyNorm, 8),
                    w = Math.Round(f.WNorm, 8),
                    h = Math.Round(f.HNorm, 8),
                    conf = Math.Round(f.Confidence, 6),
                    observed = (int)f.ObservedMask,
                    dt = Math.Round(f.DtSeconds, 6),
                    age = Math.Round(f.AgeSeconds, 6),
                }).ToArray(),
            };

            string payload = JsonSerializer.Serialize(entry, JsonOptions);
            if (EnqueueAppendLocked(Path.Combine(_outputDir, "gru_sequences.jsonl"), payload))
                Interlocked.Increment(ref _totalTracksFlushed);
        }

        private bool EnqueueAppendLocked(string path, string payload)
        {
            if (string.IsNullOrWhiteSpace(payload))
                return true;

            if (_writerChannel.Writer.TryWrite(new AppendLinesItem(path, payload)))
                return true;

            Interlocked.Increment(ref _droppedWriteBatches);
            SetLastError("Training logger writer queue is full; a training-data batch was dropped.");
            LogManager.Log(LogLevel.Warning, LastError ?? "Training logger writer queue is full.", true, 5000);
            return false;
        }

        private TaskCompletionSource<bool>? EnqueueFlushLocked(string outputDir)
        {
            var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            try
            {
                // Stop/flush is a control-path operation, not the inference hot path. It is allowed
                // to wait for queue capacity so a session boundary can never be silently skipped.
                _writerChannel.Writer
                    .WriteAsync(new FlushWritersItem(outputDir, completion))
                    .AsTask()
                    .GetAwaiter()
                    .GetResult();
            }
            catch (Exception ex)
            {
                SetLastError(ex.Message);
                completion.TrySetException(ex);
            }
            return completion;
        }

        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            WriteIndented = false,
            DefaultIgnoreCondition = JsonIgnoreCondition.Never,
        };

        private void WriteSessionMetadataLocked()
        {
            if (_outputDir == null)
                return;

            var metadata = new
            {
                schema_version = 2,
                session_id = Path.GetFileName(_outputDir),
                started_utc = DateTime.UtcNow,
                temporal_feature_schema = NeuralFeatureSchemas.TemporalV2,
                calibration_feature_schema = NeuralFeatureSchemas.CalibrationV2,
                record_format = "jsonl",
            };
            WriteJsonAtomic(Path.Combine(_outputDir, "session.json"), metadata);
        }

        private async Task WriterLoopAsync(CancellationToken cancellationToken)
        {
            var writers = new Dictionary<string, StreamWriter>(StringComparer.OrdinalIgnoreCase);
            try
            {
                await foreach (LogWriteItem item in _writerChannel.Reader.ReadAllAsync(cancellationToken))
                {
                    try
                    {
                        switch (item)
                        {
                            case AppendLinesItem append:
                                if (!writers.TryGetValue(append.Path, out StreamWriter? appendWriter))
                                {
                                    Directory.CreateDirectory(Path.GetDirectoryName(append.Path) ?? ".");
                                    appendWriter = new StreamWriter(new FileStream(
                                        append.Path,
                                        FileMode.Append,
                                        FileAccess.Write,
                                        FileShare.Read,
                                        bufferSize: 64 * 1024,
                                        useAsync: true));
                                    writers.Add(append.Path, appendWriter);
                                }
                                await appendWriter.WriteLineAsync(append.Payload).ConfigureAwait(false);
                                break;

                            case FlushWritersItem flush:
                                foreach (var (path, writer) in writers.ToList())
                                {
                                    if (!path.StartsWith(flush.OutputDir, StringComparison.OrdinalIgnoreCase))
                                        continue;
                                    await writer.FlushAsync(cancellationToken).ConfigureAwait(false);
                                    await writer.DisposeAsync().ConfigureAwait(false);
                                    writers.Remove(path);
                                }
                                flush.Completion.TrySetResult(true);
                                break;
                        }
                        SetLastError(null);
                    }
                    catch (Exception ex)
                    {
                        SetLastError(ex.Message);
                        if (item is FlushWritersItem failedFlush)
                            failedFlush.Completion.TrySetException(ex);
                        LogManager.Log(LogLevel.Warning, $"Training logger write failed: {ex.Message}");
                    }
                }
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
            }
            finally
            {
                foreach (StreamWriter writer in writers.Values)
                {
                    try
                    {
                        await writer.FlushAsync().ConfigureAwait(false);
                        await writer.DisposeAsync().ConfigureAwait(false);
                    }
                    catch
                    {
                    }
                }
            }
        }

        private static void WaitForFlush(TaskCompletionSource<bool>? completion)
        {
            if (completion == null)
                return;

            try
            {
                completion.Task.GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Warning, $"Training logger flush failed: {ex.Message}");
            }
        }

        private static void WriteJsonAtomic<T>(string path, T value)
        {
            string tempPath = path + ".tmp";
            string backupPath = path + ".bak";
            File.WriteAllText(tempPath, JsonSerializer.Serialize(value, JsonOptions));
            if (File.Exists(path))
                File.Copy(path, backupPath, overwrite: true);
            File.Move(tempPath, path, overwrite: true);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(TrackLogger));
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            if (IsActive)
                StopSession();

            _disposed = true;
            _writerChannel.Writer.TryComplete();
            try
            {
                _writerTask.GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                SetLastError(ex.Message);
                _writerCancellation.Cancel();
            }
            _writerCancellation.Dispose();
        }

        private abstract record LogWriteItem;
        private sealed record AppendLinesItem(string Path, string Payload) : LogWriteItem;
        private sealed record FlushWritersItem(string OutputDir, TaskCompletionSource<bool> Completion) : LogWriteItem;

        private sealed class LoggedTrackState
        {
            public LoggedTrackState(int trackId, string className)
            {
                TrackId = trackId;
                ClassName = className;
            }

            public int TrackId { get; }
            public string ClassName { get; set; }
            public DateTime LastObservationTimestamp { get; set; }
            public List<TrackObservation> Frames { get; } = new();
        }

        private sealed class GruSequenceEntry
        {
            public int track_id { get; set; }
            public string class_name { get; set; } = "";
            public GruFrame[] frames { get; set; } = Array.Empty<GruFrame>();
        }

        private sealed class GruFrame
        {
            public double cx { get; set; }
            public double cy { get; set; }
            public double w { get; set; }
            public double h { get; set; }
            public double conf { get; set; }
            public int observed { get; set; }
            public double dt { get; set; }
            public double age { get; set; }
        }

        private sealed class CalibrationEntry
        {
            public long frame_id { get; set; }
            public int detection_index { get; set; }
            public double raw_conf { get; set; }
            public double w_norm { get; set; }
            public double h_norm { get; set; }
            public double cx_norm { get; set; }
            public double cy_norm { get; set; }
            public double frame_age_ms { get; set; }
            public double pose_quality { get; set; }
            public int? label { get; set; }
        }
    }

    public readonly struct CalibrationSampleInput
    {
        public long FrameId { get; init; }
        public int DetectionIndex { get; init; }
        public float RawConf { get; init; }
        public float WNorm { get; init; }
        public float HNorm { get; init; }
        public float CxNorm { get; init; }
        public float CyNorm { get; init; }
        public float FrameAgeMs { get; init; }
        public float PoseQuality { get; init; }
    }
}
