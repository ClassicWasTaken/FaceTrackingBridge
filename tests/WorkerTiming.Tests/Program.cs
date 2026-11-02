// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0

using BabbleFrameBridge.Worker;

var checks = 0;
void Check(bool condition, string description)
{
    if (!condition) throw new Exception(description);
    checks++;
}

var calls = new List<string>();
var warnings = new List<string>();
uint Begin(uint period) { calls.Add($"begin:{period}"); return 0; }
uint End(uint period) { calls.Add($"end:{period}"); return 0; }

using (var unused = new ModuleTimerResolution(true, Begin, End, warnings.Add))
    Check(!unused.Active && calls.Count == 0, "Creating an inactive coordinator scope requested timer resolution.");
Check(calls.Count == 0, "Disposing an unused scope called a native timer API.");

using (var unsupported = new ModuleTimerResolution(false, Begin, End, warnings.Add))
{
    unsupported.Start(); unsupported.Start();
    Check(!unsupported.Active && calls.Count == 0, "A non-Windows scope called a Windows timer API.");
}
Check(calls.Count == 0, "Non-Windows cleanup called a Windows timer API.");

var success = new ModuleTimerResolution(true, Begin, End, warnings.Add);
success.Start(); success.Start();
Check(success.Active && calls.SequenceEqual(["begin:1"]), "A module did not acquire exactly one 1 ms request.");
success.Dispose(); success.Dispose(); success.Start();
Check(!success.Active && calls.SequenceEqual(["begin:1", "end:1"]), "Successful acquisition was not released exactly once at the matching period.");
Check(warnings.Count == 0, "A successful timer lifecycle generated a warning.");

calls.Clear();
var disposedBeforeUse = new ModuleTimerResolution(true, Begin, End, warnings.Add);
disposedBeforeUse.Dispose(); disposedBeforeUse.Start();
Check(calls.Count == 0 && !disposedBeforeUse.Active, "A disposed scope acquired a new timer request.");

using (var rejected = new ModuleTimerResolution(true,
    period => { calls.Add($"reject:{period}"); return 97; }, End, warnings.Add))
{
    rejected.Start(); rejected.Start();
    Check(!rejected.Active && warnings.Count == 1 && warnings[0].Contains("97"), "A native timer rejection did not report the fallback.");
}
Check(calls.SequenceEqual(["reject:1"]), "A rejected request was retried or released despite not being acquired.");

calls.Clear(); warnings.Clear();
using (var missing = new ModuleTimerResolution(true,
    _ => throw new DllNotFoundException("test native API unavailable"), End, warnings.Add))
{
    missing.Start();
    Check(!missing.Active && warnings.Count == 1, "An unavailable timer API prevented graceful fallback.");
}
Check(calls.Count == 0, "An unavailable begin API caused an unbalanced end call.");

calls.Clear(); warnings.Clear();
var rejectedRelease = new ModuleTimerResolution(true, Begin,
    period => { calls.Add($"end-reject:{period}"); return 13; }, warnings.Add);
rejectedRelease.Start(); rejectedRelease.Dispose(); rejectedRelease.Dispose();
Check(calls.SequenceEqual(["begin:1", "end-reject:1"]) && warnings.Count == 1 && warnings[0].Contains("13"),
    "A release error was hidden or caused duplicate release calls.");

calls.Clear(); warnings.Clear();
var missingRelease = new ModuleTimerResolution(true, Begin,
    period => { calls.Add($"end-missing:{period}"); throw new EntryPointNotFoundException("test end API unavailable"); }, warnings.Add);
missingRelease.Start(); missingRelease.Dispose(); missingRelease.Dispose();
Check(calls.SequenceEqual(["begin:1", "end-missing:1"]) && warnings.Count == 1,
    "An unavailable end API broke cleanup or was called repeatedly.");

using (var brokenLog = new ModuleTimerResolution(true, _ => 1, _ => throw new Exception("No matching acquisition"),
    _ => throw new IOException("test closed diagnostic pipe"))) brokenLog.Start();
checks++; // Failed begin + failed logging must still return normally.
using (var brokenLog = new ModuleTimerResolution(true, _ => 0, _ => 1,
    _ => throw new IOException("test closed diagnostic pipe"))) brokenLog.Start();
checks++; // Failed release + failed logging must not mask module teardown.

foreach (var failureStage in new[] { "none", "initialize", "update", "teardown", "cancel" })
{
    calls.Clear();
    var failed = false;
    try
    {
        // The host's using scope encloses its module-teardown finally.
        using var timing = new ModuleTimerResolution(true, Begin, End);
        try
        {
            timing.Start(); calls.Add("initialize");
            if (failureStage == "initialize") throw new InvalidOperationException("test initialize failure");
            calls.Add("update");
            if (failureStage == "update") throw new InvalidOperationException("test update failure");
            if (failureStage == "cancel") throw new OperationCanceledException("test cancellation");
        }
        finally
        {
            calls.Add("teardown");
            if (failureStage == "teardown") throw new InvalidOperationException("test teardown failure");
        }
    }
    catch (Exception error) when (error is InvalidOperationException or OperationCanceledException) { failed = true; }
    Check(failed == (failureStage != "none"), $"The timer scope changed the {failureStage} module outcome.");
    Check(calls[0] == "begin:1" && calls[^2] == "teardown" && calls[^1] == "end:1"
        && calls.Count(call => call == "begin:1") == 1 && calls.Count(call => call == "end:1") == 1,
        $"The {failureStage} lifecycle did not release its request once after module teardown.");
}

var begins = 0; var ends = 0;
var concurrent = new ModuleTimerResolution(true,
    _ => { Interlocked.Increment(ref begins); return 0; },
    _ => { Interlocked.Increment(ref ends); return 0; });
await Task.WhenAll(Enumerable.Range(0, 32).Select(_ => Task.Run(() => { concurrent.Start(); concurrent.Dispose(); })));
concurrent.Dispose();
Check(begins <= 1 && ends == begins && !concurrent.Active, "Concurrent lifetime callbacks leaked or duplicated a native timer request.");

Console.WriteLine($"PASS {checks} module timer lifetime checks; no native Windows timer APIs were called by this test.");
