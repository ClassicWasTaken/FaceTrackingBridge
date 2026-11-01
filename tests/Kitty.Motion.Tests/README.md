# Procedural motion checks

```powershell
dotnet run --project tests/Kitty.Motion.Tests -c Release
```

The controller is pure math; this harness creates no window, profile, network
connection, or tracking module. It checks finite and bounded generated poses,
different plans for the same skill, simultaneous running and hopping,
autonomous discovery from two basic skills, prototype changes after positive
contact, feedback limits, safe cancellation, pause, persistence, reset and
deterministic seeded behavior. The project references the app for the shared
behavior enum and requires the Windows .NET 10 development runtime.

These checks verify the motion controller. The app's separate appearance
checks verify its visual integration and carry/perch restrictions.
