// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0

using System.IO;
using System.Text.Json;

namespace BabbleFrameBridge.App;

/// <summary>Bounded local traits and normalized model observations, never raw pointer coordinates.</summary>
public sealed record KittyMemory(int Schema = 1, double Affection = .55, double Trust = .65,
    double Habituation = 0, double PlayMemory = 0, double RestMemory = 0,
    double ContactMemory = 0, double SocialNeed = .4, double[]? Preferences = null,
    int[]? Context = null, DateTimeOffset? SavedAtUtc = null,
    long GentleContacts = 0, long Pokes = 0, long Sessions = 0,
    int IdentitySeed = 0, double Playfulness = .6, double Sociability = .6, double Caution = .5,
    double Bond = 0, double PositiveContactSeconds = 0, KittyCoreMemory[]? CoreMemories = null,
    double[]? PolicyBiases = null, long LearningUpdates = 0,
    double[]? MotionPreferences = null, double[]? MotionMastery = null,
    double[]? MotionPrototypes = null, long[]? MotionPractice = null, int MotionKnownMask = 34);

public enum KittyCoreMemoryKind { FirstGentlePet, CalmPickup, SharedPlay, ComfortAfterStartle, Reunion }
public sealed record KittyCoreMemory(KittyCoreMemoryKind Kind, double Strength = .3, int Reinforcements = 1,
    DateTimeOffset? FirstUtc = null, DateTimeOffset? LastUtc = null);

/// <summary>Small atomic companion save. Loading never creates or changes a file.</summary>
public sealed class KittyMemoryStore
{
    public const int MaximumBytes = 16384, MaximumObservations = 32, ObservationWidth = 22;
    public const long MaximumCounter = 1_000_000_000;
    private static readonly JsonSerializerOptions Json = new() { MaxDepth = 8 };
    private readonly string _root;
    private readonly Func<DateTimeOffset> _utcNow;
    public string FilePath { get; }
    public string BackupPath => FilePath + ".bak";
    public string UnreadablePath => FilePath + ".unreadable";
    public string? LastError { get; private set; }
    public bool LoadedExisting { get; private set; }
    public bool RecoveredFromBackup { get; private set; }

    public KittyMemoryStore(string root, Func<DateTimeOffset>? utcNow = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        _root = Path.GetFullPath(root);
        FilePath = Path.Combine(_root, "kitty-memory.json");
        _utcNow = utcNow ?? (() => DateTimeOffset.UtcNow);
    }

    public KittyMemory Load()
    {
        LoadedExisting = RecoveredFromBackup = false; LastError = null;
        try
        {
            EnsureSafePath();
            var failed = false;
            foreach (var path in new[] { FilePath, BackupPath })
            {
                if (!File.Exists(path)) continue;
                try
                {
                    var memory = Decode(ReadBounded(path), _utcNow());
                    LoadedExisting = true; RecoveredFromBackup = path == BackupPath;
                    return memory;
                }
                catch (Exception error) when (IsStorageError(error)) { failed = true; }
            }
            if (!failed) return new();
            throw new InvalidDataException("No valid kitty memory could be read.");
        }
        catch (Exception error) when (IsStorageError(error))
        {
            LastError = "The saved kitty memory could not be read. Relay can start with fresh memories.";
            return new();
        }
    }

    public bool Save(KittyMemory memory)
    {
        ArgumentNullException.ThrowIfNull(memory);
        string? temporary = null;
        try
        {
            EnsureSafePath();
            var now = _utcNow();
            var normalized = Normalize(memory with { SavedAtUtc = now }, now);
            var bytes = JsonSerializer.SerializeToUtf8Bytes(normalized, Json);
            if (bytes.Length > MaximumBytes) throw new InvalidDataException("The kitty save exceeds its size limit.");
            Directory.CreateDirectory(_root);
            EnsureSafePath();
            temporary = Path.Combine(_root, "kitty-memory." + Guid.NewGuid().ToString("N") + ".tmp");
            using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                output.Write(bytes);
                output.Flush(flushToDisk: true);
            }
            EnsureSafePath();
            PreservePreviousSave(now);
            File.Move(temporary, FilePath, overwrite: true);
            temporary = null;
            LoadedExisting = true; RecoveredFromBackup = false; LastError = null;
            return true;
        }
        catch (Exception error) when (IsStorageError(error))
        {
            LastError = "Kitty memory could not be saved. This session can continue.";
            return false;
        }
        finally
        {
            if (temporary is not null)
                try { File.Delete(temporary); } catch (Exception error) when (IsStorageError(error)) { }
        }
    }

    public bool Forget()
    {
        try
        {
            EnsureSafePath();
            foreach (var path in new[] { FilePath, BackupPath, UnreadablePath })
            {
                if (Directory.Exists(path)) throw new IOException("A memory filename is a directory.");
                if (File.Exists(path)) File.Delete(path);
            }
            if (Directory.Exists(_root))
                foreach (var pending in Directory.EnumerateFiles(_root, "kitty-memory.*.tmp", SearchOption.TopDirectoryOnly))
                {
                    var name = Path.GetFileName(pending);
                    if (name.Length != 49 || !Guid.TryParseExact(name[13..^4], "N", out _)) continue;
                    if ((File.GetAttributes(pending) & FileAttributes.ReparsePoint) != 0)
                        throw new IOException("Kitty temporary memory cannot be a filesystem link.");
                    File.Delete(pending);
                }
            LoadedExisting = RecoveredFromBackup = false; LastError = null;
            return true;
        }
        catch (Exception error) when (IsStorageError(error))
        {
            LastError = "Kitty memory could not be cleared. Check access to the app data folder.";
            return false;
        }
    }

    public static KittyMemory Normalize(KittyMemory memory, DateTimeOffset now)
    {
        if (memory.Schema != 1) throw new InvalidDataException("Unsupported kitty memory schema.");
        static double Unit(double value, double fallback) => double.IsFinite(value) ? Math.Clamp(value, 0, 1) : fallback;
        var preferences = memory.Preferences is { Length: 35 } values
            ? values.Select(value => double.IsFinite(value) ? Math.Clamp(value, .5, 1.5) : 1).ToArray() : null;
        var age = memory.SavedAtUtc is { } saved ? now - saved : TimeSpan.MaxValue;
        var context = age >= TimeSpan.FromMinutes(-5) && age <= TimeSpan.FromMinutes(30)
            && memory.Context is { Length: > 0 and <= MaximumObservations * ObservationWidth } frames
            && frames.Length % ObservationWidth == 0 && frames.All(value => value is >= 0 and <= 65535)
            ? frames.ToArray() : null;
        var motionValid = memory.MotionPreferences is { Length: 6 } motionPreferences && motionPreferences.All(double.IsFinite)
            && memory.MotionMastery is { Length: 6 } motionMastery && motionMastery.All(double.IsFinite)
            && memory.MotionPrototypes is { Length: 30 } motionPrototypes && motionPrototypes.All(double.IsFinite)
            && memory.MotionPractice is { Length: 6 } && memory.MotionKnownMask is >= 0 and <= 63;
        return memory with
        {
            Affection = Unit(memory.Affection, .55), Trust = Unit(memory.Trust, .65),
            Habituation = Unit(memory.Habituation, 0), PlayMemory = Unit(memory.PlayMemory, 0),
            RestMemory = Unit(memory.RestMemory, 0), ContactMemory = Unit(memory.ContactMemory, 0),
            SocialNeed = Unit(memory.SocialNeed, .4), Preferences = preferences, Context = context,
            GentleContacts = Math.Clamp(memory.GentleContacts, 0, MaximumCounter),
            Pokes = Math.Clamp(memory.Pokes, 0, MaximumCounter), Sessions = Math.Clamp(memory.Sessions, 0, MaximumCounter),
            Playfulness = Unit(memory.Playfulness, .6), Sociability = Unit(memory.Sociability, .6),
            Caution = Unit(memory.Caution, .5), Bond = Unit(memory.Bond, 0),
            PolicyBiases = memory.PolicyBiases is { Length: 40 } biases
                ? biases.Select(value => double.IsFinite(value) ? Math.Clamp(value, -.8, .8) : 0).ToArray() : null,
            LearningUpdates = Math.Clamp(memory.LearningUpdates, 0, MaximumCounter),
            MotionPreferences = motionValid ? memory.MotionPreferences!.Select(value => Math.Clamp(value, 0, 1)).ToArray() : null,
            MotionMastery = motionValid ? memory.MotionMastery!.Select(value => Math.Clamp(value, 0, 1)).ToArray() : null,
            MotionPrototypes = motionValid ? memory.MotionPrototypes!.Select(value => Math.Clamp(value, 0, 1)).ToArray() : null,
            MotionPractice = motionValid ? memory.MotionPractice!.Select(value => Math.Clamp(value, 0L, 1_000_000L)).ToArray() : null,
            MotionKnownMask = motionValid ? memory.MotionKnownMask : 34,
            PositiveContactSeconds = double.IsFinite(memory.PositiveContactSeconds) ? Math.Clamp(memory.PositiveContactSeconds, 0, MaximumCounter) : 0,
            CoreMemories = (memory.CoreMemories ?? []).Where(core => core is not null && Enum.IsDefined(core.Kind))
                .Take(16).GroupBy(core => core.Kind).Select(group =>
                {
                    var core = group.OrderByDescending(item => double.IsFinite(item.Strength) ? item.Strength : 0).First();
                    var first = core.FirstUtc is { } a && a <= now.AddMinutes(5) && a.Year >= 2020 ? core.FirstUtc : null;
                    var last = core.LastUtc is { } b && b <= now.AddMinutes(5) && b.Year >= 2020 ? core.LastUtc : null;
                    if (first > last) (first, last) = (last, first);
                    return core with { Strength = Unit(core.Strength, .3), Reinforcements = Math.Clamp(core.Reinforcements, 1, 1_000_000), FirstUtc = first, LastUtc = last };
                }).ToArray()
        };
    }

    private void EnsureSafePath()
    {
        foreach (var path in new[] { _root, FilePath, BackupPath, UnreadablePath })
            if ((File.Exists(path) || Directory.Exists(path)) && (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0)
                throw new IOException("Kitty memory paths cannot be filesystem links.");
    }

    private static byte[] ReadBounded(string path, int minimumLength = 2)
    {
        using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (input.Length < minimumLength || input.Length > MaximumBytes) throw new InvalidDataException("The kitty save exceeds its size limit.");
        var bytes = new byte[(int)input.Length]; input.ReadExactly(bytes); return bytes;
    }

    private static KittyMemory Decode(byte[] bytes, DateTimeOffset now) => Normalize(
        JsonSerializer.Deserialize<KittyMemory>(bytes, Json) ?? throw new InvalidDataException("The kitty save was empty."), now);

    private void PreservePreviousSave(DateTimeOffset now)
    {
        if (!File.Exists(FilePath)) return;
        // A damaged/future profile is kept verbatim before replacing it. An
        // oversized unknown file is left untouched and saving fails safely.
        // Preserve incomplete 0/1-byte writes too: they are invalid to load,
        // but must not prevent a recovered backup from saving new memories.
        var previous = ReadBounded(FilePath, minimumLength: 0);
        var destination = BackupPath;
        try { _ = Decode(previous, now); }
        catch (Exception error) when (IsStorageError(error)) { destination = UnreadablePath; }
        var temporary = Path.Combine(_root, "kitty-memory." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            { output.Write(previous); output.Flush(flushToDisk: true); }
            EnsureSafePath();
            File.Move(temporary, destination, overwrite: true);
        }
        finally
        {
            try { File.Delete(temporary); } catch (Exception error) when (IsStorageError(error)) { }
        }
    }

    private static bool IsStorageError(Exception error) => error is IOException or InvalidDataException or UnauthorizedAccessException
        or JsonException or NotSupportedException or ArgumentException;
}

/// <summary>Slow strokes across the cat; hovering, rapid waving and pressed-button drags are distinct.</summary>
public sealed class KittyPetGesture
{
    private double _duration, _distance;
    public void Reset() { _duration = _distance = 0; }
    public bool Sample(double seconds, double distance, bool overCat, bool buttonPressed)
    {
        if (!double.IsFinite(seconds) || !double.IsFinite(distance) || seconds is <= 0 or > .2
            || distance < 0 || !overCat || buttonPressed) { Reset(); return false; }
        var speed = distance / seconds;
        if (speed is < 12 or > 300) { Reset(); return false; }
        _duration += seconds; _distance += distance;
        if (_duration < .3 || _distance < 12) return false;
        Reset(); return true;
    }
}
