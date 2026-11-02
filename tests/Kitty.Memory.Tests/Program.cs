// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0
using System.Text.Json;
using BabbleFrameBridge.App;

var checks = 0;
void Check(bool value, string message) { if (!value) throw new InvalidOperationException(message); checks++; }
var root = Path.Combine(Path.GetTempPath(), "FaceTrackingBridge-KittyMemory-" + Guid.NewGuid().ToString("N"));
var now = new DateTimeOffset(2026, 10, 31, 18, 0, 0, TimeSpan.Zero);
var clock = now;
var store = new KittyMemoryStore(root, () => clock);
Check(!Directory.Exists(root), "Constructing the store wrote data.");
Check(store.Load() == new KittyMemory() && !Directory.Exists(root), "Loading a new disabled kitty created data.");
Check(store.Forget() && !Directory.Exists(root), "Forgetting an unused companion created data or failed.");
var packed = Enumerable.Range(0, 704).Select(i => i * 93 % 65536).ToArray();
var core = new KittyCoreMemory(KittyCoreMemoryKind.FirstGentlePet, .7, 4, now.AddDays(-2), now);
var memory = new KittyMemory(Affection: .78, Trust: .82, Habituation: .24, Preferences: Enumerable.Repeat(1.2, 35).ToArray(),
    Context: packed, GentleContacts: 37, Pokes: 4, Sessions: 3, IdentitySeed: 123456, Playfulness: .8,
    Sociability: .7, Caution: .2, Bond: .6, PositiveContactSeconds: 120, CoreMemories: [core], PolicyBiases: Enumerable.Range(0, 40).Select(index => (index - 20) / 40d).ToArray(),
    LearningUpdates: 123, MotionPreferences: Enumerable.Repeat(.6, 6).ToArray(), MotionMastery: Enumerable.Repeat(.4, 6).ToArray(),
    MotionPrototypes: Enumerable.Repeat(.3, 30).ToArray(), MotionPractice: Enumerable.Repeat(21L, 6).ToArray(), MotionKnownMask: 63);
Check(store.Save(memory), "A valid memory failed to save.");
Check(new FileInfo(store.FilePath).Length <= KittyMemoryStore.MaximumBytes, "A maximum context exceeded the hard file budget.");
var firstBytes = File.ReadAllBytes(store.FilePath);
var restored = new KittyMemoryStore(root, () => clock).Load();
Check(restored.Trust == .82 && restored.Affection == .78 && restored.Sessions == 3 && restored.GentleContacts == 37, "Aggregate traits or counters were lost.");
Check(restored.IdentitySeed == 123456 && restored.Playfulness == .8 && restored.Bond == .6, "Individual personality or bond was lost.");
Check(restored.CoreMemories is { Length: 1 } && restored.CoreMemories[0] == core, "A positive core memory was not retained.");
Check(restored.Context!.SequenceEqual(packed) && restored.Preferences!.SequenceEqual(memory.Preferences!), "Context or preferences did not round-trip.");
Check(restored.SavedAtUtc == now, "Saving did not record its UTC time.");
Check(restored.PolicyBiases!.SequenceEqual(memory.PolicyBiases!) && restored.LearningUpdates == 123, "Actual learned policy weights/count were not persisted.");
Check(restored.MotionPreferences!.SequenceEqual(memory.MotionPreferences!) && restored.MotionPrototypes!.Length == 30
    && restored.MotionPractice!.All(value => value == 21) && restored.MotionKnownMask == 63, "Procedural skill learning did not round-trip.");

Check(File.ReadAllBytes(store.FilePath).SequenceEqual(firstBytes), "Reading rewrote the save.");
clock = now.AddMinutes(31);
var stale = store.Load();
Check(stale.Context is null && stale.Trust == .82 && stale.CoreMemories is { Length: 1 }, "Old context replayed stale stimuli or discarded lasting traits.");
Check(File.ReadAllBytes(store.FilePath).SequenceEqual(firstBytes), "Expiring old context modified the saved file during read.");
clock = now.AddMinutes(-10);
Check(store.Load().Context is null, "A save from far in the future restored context.");
clock = now;
var invalid = memory with { Schema = 100 };
Check(!store.Save(invalid) && File.ReadAllBytes(store.FilePath).SequenceEqual(firstBytes), "A failed schema save replaced the previous valid save.");
Check(!Directory.GetFiles(root, "*.tmp").Any(), "Failed save left a temporary file.");

var bounded = KittyMemoryStore.Normalize(memory with
{
    Affection = double.NaN, Trust = 9, Habituation = -2, PositiveContactSeconds = double.PositiveInfinity,
    GentleContacts = long.MaxValue, Pokes = -99, Sessions = long.MaxValue,
    Preferences = Enumerable.Repeat(100d, 35).ToArray(), Context = [-1, 5], SavedAtUtc = now,
    CoreMemories = [core, core with { Strength = 1.5 }, new((KittyCoreMemoryKind)999, 1), new(KittyCoreMemoryKind.Reunion, double.NaN, -2, now.AddYears(5), now.AddYears(5))]
}, now);
Check(bounded.Affection == .55 && bounded.Trust == 1 && bounded.Habituation == 0, "Invalid traits were not finite/clamped.");
Check(bounded.GentleContacts == KittyMemoryStore.MaximumCounter && bounded.Pokes == 0 && bounded.PositiveContactSeconds == 0, "Counters or contact time escaped bounds.");
Check(bounded.Preferences!.All(value => value == 1.5) && bounded.Context is null, "Bad preference/context dimensions were accepted.");
Check(bounded.CoreMemories is { Length: 2 } && bounded.CoreMemories[0].Strength == 1, "Core-memory validation failed enum/duplicate/strength bounds.");
Check(bounded.CoreMemories![1].Reinforcements == 1 && bounded.CoreMemories[1].FirstUtc is null, "Invalid core timestamps/reinforcements were accepted.");
var biasBounded = KittyMemoryStore.Normalize(memory with { PolicyBiases = Enumerable.Repeat(99d, 40).ToArray(), LearningUpdates = long.MaxValue }, now);
Check(biasBounded.PolicyBiases!.All(value => value == .8) && biasBounded.LearningUpdates == KittyMemoryStore.MaximumCounter,
    "Saved online learning escaped its weight/count limits.");
var biasInvalid = KittyMemoryStore.Normalize(memory with { PolicyBiases = Enumerable.Repeat(double.NaN, 40).ToArray() }, now);
Check(biasInvalid.PolicyBiases!.All(value => value == 0), "Nonfinite learned biases were retained.");
Check(KittyMemoryStore.Normalize(memory with { PolicyBiases = [1] }, now).PolicyBiases is null, "A wrong-size adapter was accepted.");
var motionBounded = KittyMemoryStore.Normalize(memory with { MotionPreferences = Enumerable.Repeat(99d, 6).ToArray(), MotionPractice = Enumerable.Repeat(long.MaxValue, 6).ToArray() }, now);
Check(motionBounded.MotionPreferences!.All(value => value == 1) && motionBounded.MotionPractice!.All(value => value == 1_000_000),
    "Motion learning escaped its practical range.");
var motionInvalid = KittyMemoryStore.Normalize(memory with { MotionMastery = [1] }, now);
Check(motionInvalid.MotionPreferences is null && motionInvalid.MotionMastery is null && motionInvalid.MotionPrototypes is null
    && motionInvalid.MotionPractice is null && motionInvalid.MotionKnownMask == 34 && motionInvalid.Trust == memory.Trust
    && motionInvalid.PolicyBiases!.SequenceEqual(memory.PolicyBiases!), "A corrupt skill record damaged unrelated personal memories or partially restored invalid motion.");
var oldCompatible = KittyMemoryStore.Normalize(new(Trust: .8), now);
Check(oldCompatible.Trust == .8 && oldCompatible.PolicyBiases is null && oldCompatible.LearningUpdates == 0
    && oldCompatible.MotionKnownMask == 34, "An old profile could not acquire neutral new learning fields.");
var huge = memory with { Context = Enumerable.Repeat(65535, 22 * 10000).ToArray(), CoreMemories = Enumerable.Repeat(core, 10000).ToArray(), SavedAtUtc = now };
var limited = KittyMemoryStore.Normalize(huge, now);
Check(limited.Context is null && limited.CoreMemories!.Length == 1, "Unbounded arrays escaped validation.");
Check(store.Save(huge) && new FileInfo(store.FilePath).Length <= KittyMemoryStore.MaximumBytes, "Oversized in-memory arrays made an oversized disk save.");
if (File.Exists(store.BackupPath)) File.Delete(store.BackupPath);

foreach (var contents in new[] { "", "{", "{broken", "null", "{\"Schema\":2}", new string(' ', KittyMemoryStore.MaximumBytes + 1) })
{
    File.WriteAllText(store.FilePath, contents);
    var loaded = store.Load();
    Check(loaded == new KittyMemory() && !store.LoadedExisting && store.LastError is not null, "A corrupt, future or oversized save did not fail safely.");
    Check(File.ReadAllText(store.FilePath) == contents, "A failed load modified the corrupt save.");
}
File.WriteAllText(store.FilePath, "{\"Schema\":1,\"Trust\":0.75}");
Check(store.Load().Trust == .75, "An older minimal compatible profile could not load.");
Check(store.Save(memory), "A valid session could not replace a corrupt save safely.");
using (var locked = new FileStream(store.FilePath, FileMode.Open, FileAccess.Read, FileShare.None))
{
    Check(!store.Save(memory with { Trust = .1 }), "A locked save was overwritten.");
    _ = store.Load();
    Check(store.LoadedExisting && store.RecoveredFromBackup, "A locked primary save could not recover its valid previous profile.");
}
Check(store.Load().Trust == memory.Trust, "A failed locked write damaged the prior save.");
var otherRoot = root + "-other";
Check(new KittyMemoryStore(otherRoot).Load().Trust == .65 && !Directory.Exists(otherRoot), "Independent profiles share or create memory.");
var pendingFile = Path.Combine(root, "kitty-memory." + Guid.NewGuid().ToString("N") + ".tmp");
var unrelatedFile = Path.Combine(root, "kitty-memory.other.tmp");
File.WriteAllText(pendingFile, "unfinished companion memory"); File.WriteAllText(unrelatedFile, "unrelated");
Check(store.Forget() && !File.Exists(store.FilePath) && !store.LoadedExisting, "Forget did not remove only the memory file.");
Check(!File.Exists(pendingFile) && File.ReadAllText(unrelatedFile) == "unrelated", "Forget failed to retire owned incomplete saves or removed an unrelated file.");
Check(Directory.Exists(root), "Forget removed the app-data directory.");

var recovery = new KittyMemoryStore(root + "-recovery", () => now);
Check(recovery.Save(memory) && recovery.Save(memory with { Trust = .9 }), "The recovery fixture could not establish a last-good backup.");
var goodBackup = File.ReadAllBytes(recovery.BackupPath);
File.WriteAllText(recovery.FilePath, "{broken primary");
var recovered = recovery.Load();
Check(recovery.LoadedExisting && recovery.RecoveredFromBackup && recovered.Trust == memory.Trust
    && recovered.CoreMemories is { Length: 1 }, "A corrupt primary lost the lasting core memory despite a valid backup.");
Check(File.ReadAllText(recovery.FilePath) == "{broken primary" && File.ReadAllBytes(recovery.BackupPath).SequenceEqual(goodBackup),
    "Read-only backup recovery modified the disabled companion's files.");
Check(recovery.Save(recovered) && File.ReadAllText(recovery.UnreadablePath) == "{broken primary"
    && File.ReadAllBytes(recovery.BackupPath).SequenceEqual(goodBackup), "Repair overwrote a last-good backup or discarded the unreadable primary.");
foreach (var incomplete in new[] { Array.Empty<byte>(), new byte[] { (byte)'{' } })
{
    File.WriteAllBytes(recovery.FilePath, incomplete);
    var incompleteRecovered = recovery.Load();
    Check(recovery.RecoveredFromBackup && incompleteRecovered.Trust == memory.Trust,
        "An incomplete primary could not recover the last-good memory.");
    Check(File.ReadAllBytes(recovery.FilePath).SequenceEqual(incomplete)
        && File.ReadAllBytes(recovery.BackupPath).SequenceEqual(goodBackup),
        "Loading an incomplete primary modified original or recovery bytes.");
    Check(recovery.Save(incompleteRecovered with { Trust = .93 })
        && File.ReadAllBytes(recovery.UnreadablePath).SequenceEqual(incomplete)
        && File.ReadAllBytes(recovery.BackupPath).SequenceEqual(goodBackup),
        "Saving after incomplete-primary recovery failed, discarded the original, or overwrote the last-good backup.");
    Check(recovery.Load().Trust == .93 && !recovery.RecoveredFromBackup,
        "New memories did not persist after repairing an incomplete primary.");
}
const string future = "{\"Schema\":77,\"FuturePersonalBond\":\"preserve this profile\"}";
File.WriteAllText(recovery.FilePath, future);
Check(recovery.Load().CoreMemories is { Length: 1 } && recovery.RecoveredFromBackup, "A future schema prevented safe last-good fallback.");
Check(recovery.Save(memory) && File.ReadAllText(recovery.UnreadablePath) == future, "Saving over a future schema silently discarded it.");
var oversizedUnknown = new string('x', KittyMemoryStore.MaximumBytes + 1);
File.WriteAllText(recovery.FilePath, oversizedUnknown);
Check(!recovery.Save(memory) && File.ReadAllText(recovery.FilePath) == oversizedUnknown, "An oversized unknown profile was silently overwritten.");
Check(recovery.Forget() && !File.Exists(recovery.FilePath) && !File.Exists(recovery.BackupPath) && !File.Exists(recovery.UnreadablePath),
    "Forget retained a recovery file that could resurrect old memories.");

var gesture = new KittyPetGesture();
for (var i = 0; i < 20; i++) Check(!gesture.Sample(.1, 0, true, false), "Stationary hovering counted as a stroke.");
for (var i = 0; i < 20; i++) Check(!gesture.Sample(.1, 50, true, false), "Rapid waving counted as gentle petting.");
for (var i = 0; i < 20; i++) Check(!gesture.Sample(.1, 5, true, true), "A drag counted as petting.");
Check(!gesture.Sample(.1, 5, true, false) && !gesture.Sample(.1, 5, true, false) && gesture.Sample(.11, 5, true, false), "A sustained slow stroke was not recognized.");
Check(!gesture.Sample(.1, 5, true, false), "Petting emitted again without a complete stroke interval.");
gesture.Sample(.1, 5, false, false);
Check(!gesture.Sample(.11, 5, true, false), "Leaving the cat retained an unfinished stroke.");
foreach (var invalidTime in new[] { double.NaN, double.PositiveInfinity, -1, 10 })
    Check(!gesture.Sample(invalidTime, 5, true, false), "Invalid timing created a pet interaction.");
Check(!gesture.Sample(.1, double.NaN, true, false), "Invalid movement created a pet interaction.");
Console.WriteLine($"PASS {checks} kitty memory/gesture checks; bounded {KittyMemoryStore.MaximumBytes:N0}-byte local save, atomic recovery, context expiry, lasting personality and semantic memories.");
