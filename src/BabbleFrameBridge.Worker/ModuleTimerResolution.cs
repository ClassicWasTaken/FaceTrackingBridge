// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0

namespace BabbleFrameBridge.Worker;

/// <summary>
/// Owns a process-local Windows timer-resolution request while an SDK module is hosted.
/// Modules retain their requested Sleep/Wait intervals; no extra update cadence is imposed.
/// </summary>
internal sealed class ModuleTimerResolution(
    bool isWindows,
    Func<uint, uint> beginPeriod,
    Func<uint, uint> endPeriod,
    Action<string>? logWarning = null) : IDisposable
{
    private const uint PeriodMilliseconds = 1;
    private readonly object gate = new();
    private bool attempted, active, disposed;

    internal bool Active { get { lock (gate) return active; } }

    // Construction is deliberately inactive: a coordinator and a worker waiting
    // for hardware must not change timer resolution before they host a module.
    internal void Start()
    {
        lock (gate)
        {
            if (attempted || disposed) return;
            attempted = true;
            if (!isWindows) return;
            try
            {
                var result = beginPeriod(PeriodMilliseconds);
                active = result == 0;
                if (!active) Warn($"Windows rejected the module timer-resolution request (code {result}). Tracking will continue with default timing.");
            }
            catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
            {
                Warn($"Module timer precision is unavailable: {error.Message}. Tracking will continue with default timing.");
            }
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            if (!active) return;
            active = false;
            try
            {
                var result = endPeriod(PeriodMilliseconds);
                if (result != 0) Warn($"Windows could not release module timer precision (code {result}).");
            }
            catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
            {
                Warn($"Could not release module timer precision: {error.Message}");
            }
        }
    }

    private void Warn(string message)
    {
        // A closed diagnostic pipe must not prevent tracking or timer cleanup.
        try { logWarning?.Invoke(message); } catch { }
    }
}
