# Changelog

## 1.1.1 - Local preview, unreleased

- Relay uses a dedicated causal Transformer trained on original, research-informed cat-behavior sequences.
- A persistent individual temperament, growing bond, shared core memories, preferences and recent context load locally across sessions. No cloud or background weight training.
- Gentle cursor strokes pet the cat; clicks poke it; dragging picks it up. Calm handling and positive contact can reinforce shared memories.
- Settings shows the remembered bond and shared moments, with a way to forget only the kitty's memories. The kitty still starts off every launch and remains completely silent.
- Memory saves are bounded and atomic. Corrupt or inaccessible kitty data does not prevent the bridge from opening.
- Reduced-motion mode keeps interactions and memory working without enabling the animation loop. Rapid pokes no longer manufacture simulated elapsed time.

## 1.1.0

Changes since **1.0.4**. This release replaces the withdrawn 1.5.0 release.

### Combine tracking modules

- Choose separate **Eyes & eyebrows** and **Face & mouth** modules, or use one module for both. Your assignments and each profile’s settings are remembered.
- Each selected module shows **Selected · Eyes**, **Selected · Face**, or **Selected** when it provides both roles.
- Separate sources run in isolated workers and feed one combined VRChat output. A shared source runs once. Each separate module initializes only its assigned role.
- Eye tracking owns gaze, eyelids, pupils, eye expressions and eyebrows; face tracking owns the remaining mouth, cheek, nose, jaw and tongue expressions.
- A failed or stale source is neutralized independently while the other source continues. Startup validates a complete frame before accepting a ready source; unavailable hardware can wait for connection. Modules without the assigned capability fail with clear guidance.
- Known input-port collisions, conflicts with the VRChat ports, and duplicate Virtual Desktop consumers are rejected with guidance. Failed source changes preserve the previous assignments and attempt to restore a running pair.

### More module downloads

- Added official **SRanipal**, **Bigscreen Beyond**, and **EyeTrackVR** downloads alongside Babble Headset Runtime, Baballonia, Steam Link and Virtual Desktop.
- Improved standalone DLL validation, redirected downloads, cancellation, incomplete-download detection, and bounded package checks. Modules remain separate downloads under their authors’ licenses.

### Optional holographic kitty

- Meet **Relay**, an original articulated robot cat with a small local recurrent behavior model, attention, moods, play, grooming, investigation and rest.
- Relay can swat the pointer, wobble panels, get the zoomies, climb visible panels, perch on edges, and be picked up and dragged.
- Spring-driven body, head, ears, paws and tail make movement fluid. Relay looks around independently of the pointer.
- Completely silent: meows, chirps and yawns appear as cat-noise text bubbles. No speech, audio, cloud service or model download.
- Click the bottom-left kitty face to enable it. It starts **off on every launch** and pauses while the app is hidden in the tray.

### Interface and reliability fixes

- Notifications use the existing header space without moving pages or controls. Showing, replacing and dismissing a notification no longer changes a button’s position.
- Connection particles smoothly slow, freeze and dim when the bridge stops, then brighten and resume from the same position when it starts.
- Fixed the **Manage modules** button flashing during Start/Stop, and made selected-module labels readable in both themes.
- Improved missing/corrupt module recovery without silently switching saved sources; invalid modules are still checked before activation.
- Improved worker shutdown and orphan cleanup, module removal/rollback, settings-save recovery, and activity logging.
- Native tracking stays connected through transient avatar-discovery failures; real avatar or capability changes still reset the route.
- The installer preserves settings and imported modules when upgrading from 1.0.4 or replacing the withdrawn 1.5.0 with 1.1.0. Other downgrade protections remain in place.

Tracking validation uses isolated profiles and synthetic inputs, including official module DLLs. Physical headset calibration and full avatar behavior still require the relevant hardware. Available expressions depend on the hardware, module and avatar.

[Download the Windows installer](https://github.com/ClassicWasTaken/FaceTrackingBridge/releases/download/v1.1.0/FaceTracking-Bridge-Setup.exe)
