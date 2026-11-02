# Baballonia latency investigation

Investigated on Windows x64, 1 November 2026, after a report of slower mouth
movement with Baballonia selected for both eyes and face, with Relay off.

## Finding

The older **3.1.0** module showed a reproducible timing difference. Its actual
release DLL caches incoming values and copies them during `Update()`, then calls
`Thread.Sleep(10)`. VRCFaceTracking requests 1 ms Windows timer resolution in
module-hosting processes. Face Tracking Bridge 1.1.1 omitted that request.
Sleep-based module updates consequently had more delay and jitter here.

A local fix requests the same timer precision immediately before a worker
initializes a module and releases it after teardown, including failure and
cancellation. Idle coordinators, the main interface, and workers waiting for
tracking software do not acquire it. Failure of the timer API logs a warning
and leaves tracking available. The output rate remains 90 Hz.

## Measured shared-module mouth response

Each row uses the same official **3.1.0** DLL and 100 randomized OSC steps,
with sequential runs and no competing benchmark. Times are milliseconds from
test input injection to receiving the first matching VRChat output packet.

| Implementation | Mean | Median | p95 |
| --- | ---: | ---: | ---: |
| Face Tracking Bridge 1.1.1 | 14.159 | 14.384 | 23.172 |
| Bridge with timer fix | 11.053 | 10.940 | 18.375 |
| Upstream VRCFaceTracking core, run 1 | 10.660 | 10.163 | 17.307 |
| Upstream VRCFaceTracking core, run 2 | 10.993 | 11.139 | 17.595 |

The fix reduced the measured average by about **3 ms** and p95 by about **5 ms**,
bringing this older-module path close to upstream's core pipeline. This does
not establish the complete cause or size of the delay on the reporting PC.

The official **3.2.0** DLL writes incoming values directly to tracking data.
Before this fix, shared-module mouth response was already about **6 ms median**
and **11 ms p95**, overlapping two upstream-core runs (6.45 ms median,
10.57–11.19 ms p95). The timer change did not produce a meaningful improvement
in this newer module's shared input path. Both legacy and native output were
tested. A further 200-step Jerry-style `FT/v2` float test measured 6.13 ms
median / 11.11 ms p95 before the change and 5.85 / 10.66 ms afterward, within
normal run variation. A separate split-module test improved from roughly 13–15 ms median to
11–12 ms with the timer request; that extra pipeline is not used when one
Baballonia profile provides both roles.

All observed 10%, 50%, and 90% step responses coincided: these paths sent the
new value immediately at their next sample, without an additional smoothing
ramp. Early exploratory legacy runs contained isolated 52–81 ms spikes; the
data supports the typical timing improvement, not a guarantee against all
operating-system scheduling stalls.

## Method and limits

- Module ZIPs, settings, avatar definitions, ports and worker processes were
  isolated. No real VRChat, SteamVR registration, imported module or kitty
  memory was modified.
- A dedicated UDP capture thread records receipt time before packet decoding.
  Avatar routing and actual output are warmed before measurement; random step
  intervals avoid a fixed phase relationship with the tracking clock.
- Steps exercise mouth and eyelid input. This is not a full-rate multi-channel
  camera stream saturation benchmark. Repeated scenarios cover shared/split
  source configuration and legacy/native output; `FT/v2` prefixed floats can
  be selected for Jerry-style routing.
- The upstream comparison uses actual upstream Core, module process, sandbox
  IPC, send coordinator, mapper and OSC sender at commit
  `e1056bb98843dd0fd1b152f24cce3b09d0d19db4`. Its copied module entry point
  disables telemetry, with tracking and scheduling code unchanged. Both sides
  disable mutations for comparison. Upstream's harness uses all-relevant
  legacy mapping; the bridge fixture binds explicit float parameters.
- This measures a **core input-to-output pipeline**, not the installed Steam
  application's interface or the friend's computer. Camera exposure, model
  inference, headset/network transport, full-rate input, WPF rendering,
  avatar animation, remote-avatar synchronization and display latency remain
  outside these results. Jerry's animator may add its own smoothing or
  quantization after the OSC packet is delivered.
- There is no extra filter enabled in the bridge's worker. A shared eye/face
  profile starts one module instance; it does not take the split-source path.

## Exact module receipts

Actual binaries were checked rather than inferring behavior from tagged source.
Exact measurements and tested binary hashes are retained in the
[machine-readable receipt](BABALLONIA-LATENCY.json).

| Official module | ZIP SHA-256 | DLL SHA-256 |
| --- | --- | --- |
| 3.1.0, rc6 asset | `e18d293769b3f04bc98b1a75f2c49e5907e32290b58e935daa99eceab3740413` | `729ac7d101f9f8b81fd83c1ca9ec953da58b52e27063e99033cc0b578e30db3c` |
| 3.2.0, Expressions Alpha asset | `1501bcf8ed1a8fb0b9d8e166ab1c95ab838c4e33d09524f2f537bf6f87d8c032` | `b85dc69b2a814d71947b4ad8d9747fd8cf50c13f335c8426a3b6a2e2ec2f8c19` |

Reproduce the current-module measurements with
[`baballonia_latency_test.py`](../tests/baballonia_latency_test.py).
For the older module, add `--legacy-eye-addresses` to select its uppercase
`/LeftEyeLid` and `/RightEyeLid` inputs instead of the current-module defaults.

References: [VRCFaceTracking timer initialization](https://github.com/benaclejames/VRCFaceTracking/blob/e1056bb98843dd0fd1b152f24cce3b09d0d19db4/VRCFaceTracking.Core/MainStandalone.cs),
[sandbox timer initialization](https://github.com/benaclejames/VRCFaceTracking/blob/e1056bb98843dd0fd1b152f24cce3b09d0d19db4/VRCFaceTracking.ModuleProcess/ModuleProcessMain.cs),
[Microsoft timeBeginPeriod](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod),
[Microsoft Sleep](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-sleep).
