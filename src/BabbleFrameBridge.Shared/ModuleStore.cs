// Copyright (c) 2026 Frame Bridge contributors.
// SPDX-License-Identifier: Apache-2.0

using System.IO.Compression;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text;
using System.Reflection;
using System.Text.Json.Serialization;

namespace BabbleFrameBridge.Shared;

public sealed record ModuleRelease(
    string Id,
    string Version,
    string ModuleId,
    string DllFileName,
    DateTimeOffset InstalledAt,
    string Sha256,
    string DirectoryPath,
    string ModuleName,
    ModuleKind Kind = ModuleKind.BabbleHeadsetRuntime,
    bool SupportsHeadset = false,
    string? AssemblyName = null)
{
    [JsonIgnore]
    public bool IsAvailable => File.Exists(Path.Combine(DirectoryPath, DllFileName));
}

/// <summary>
/// Stages packages without loading code, then switches the active release with an atomic
/// manifest replacement. Existing package directories are never edited during an update.
/// </summary>
public sealed class ModuleStore
{
    public const string BaballoniaModuleId = "360b014b-b57b-450f-8f12-9904618ff370";
    public const string BaballoniaDll = "VRCFaceTracking.Baballonia.dll";
    private const long MaxArchiveBytes = 256L * 1024 * 1024;
    private const long MaxExpandedBytes = 512L * 1024 * 1024;
    private const int MaxEntries = 4096;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true, PropertyNameCaseInsensitive = true };
    private readonly object _gate = new();
    private readonly string _storeRoot;
    private readonly string _releasesRoot;
    private readonly string _statePath;
    private StoreState _state;
    private readonly List<string> _recoveryIssues = [];
    private string? _readOnlyReason;
    private bool _preserveStateOnNextWrite;
    private const int MaximumStateBytes = 1024 * 1024, MaximumSavedModules = 4096;
    private const long MaximumPreservedStateBytes = 32L * 1024 * 1024;

    public IReadOnlyList<string> RecoveryIssues { get { lock (_gate) return _recoveryIssues.ToArray(); } }
    public bool IsReadOnly => _readOnlyReason is not null;
    public string? ReadOnlyReason => _readOnlyReason;
    public string? RecoveryNotice => _readOnlyReason ?? (_recoveryIssues.Count > 0
        ? "Some saved modules need attention. Usable modules are available; existing files have been preserved." : null);

    public ModuleStore(string root)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(root);
        _storeRoot = Path.Combine(Path.GetFullPath(root), "modules");
        _releasesRoot = Path.Combine(_storeRoot, "releases");
        _statePath = Path.Combine(_storeRoot, "module-state.json");
        try { Directory.CreateDirectory(_releasesRoot); }
        catch (Exception error) when (IsRecoveryError(error))
        {
            _readOnlyReason = "The saved module library is read-only. Existing files have been preserved.";
            AddRecoveryIssue("The module releases folder could not be opened for writing: " + error.Message);
        }
        _state = ReadState();
        if (!IsReadOnly)
            try { CleanupRemovedReleases(); }
            catch (Exception error) when (IsRecoveryError(error))
            { AddRecoveryIssue("Deferred module cleanup could not finish: " + error.Message); }
    }

    public ModuleRelease? Current { get { lock (_gate) return ReadRelease(_state.Current); } }
    public ModuleRelease? Previous { get { lock (_gate) return ReadRelease(_state.Previous); } }
    public ModuleRelease? EyeModule { get { lock (_gate) return ReadRelease(EyeId); } }
    public ModuleRelease? FaceModule { get { lock (_gate) return ReadRelease(FaceId); } }
    public ModuleRelease? PreviousEyeModule { get { lock (_gate) return ReadRelease(PreviousEyeId); } }
    public ModuleRelease? PreviousFaceModule { get { lock (_gate) return ReadRelease(PreviousFaceId); } }
    public ModuleChannelSelection Selection { get { lock (_gate) return ReadSelection(EyeId, FaceId); } }
    public ModuleChannelSelection PreviousSelection { get { lock (_gate) return ReadSelection(PreviousEyeId, PreviousFaceId); } }
    public IReadOnlyList<ModuleRelease> Installed { get { lock (_gate) return InstalledIds().Select(id => ReadRelease(id)!).OrderBy(x => x.Kind).ThenByDescending(x => x.InstalledAt).ToArray(); } }

    // A legacy selection owns both roles until the next state write upgrades it.
    // Schema 3 uses explicit nulls so a cleared selection cannot reappear via Current.
    private string? EyeId => _state.Schema < 3 ? _state.Current : _state.EyeModuleId;
    private string? FaceId => _state.Schema < 3 ? _state.Current : _state.FaceModuleId;
    private string? PreviousEyeId => _state.Schema < 3 ? _state.Previous : _state.PreviousEyeId;
    private string? PreviousFaceId => _state.Schema < 3 ? _state.Previous : _state.PreviousFaceId;
    private ModuleChannelSelection ReadSelection(string? eyeId, string? faceId)
    {
        var eye = ReadRelease(eyeId);
        return new(eye, eyeId == faceId ? eye : ReadRelease(faceId));
    }

    private string[] InstalledIds() => (_state.Installed ?? [])
        .Concat(new[] { _state.Current, _state.Previous, EyeId, FaceId, PreviousEyeId, PreviousFaceId }.OfType<string>())
        .Distinct(StringComparer.OrdinalIgnoreCase).ToArray();

    private static StoreState SelectionState(string? eyeId, string? faceId, string? previousEyeId,
        string? previousFaceId, IEnumerable<string> installed) => new(3, eyeId, previousEyeId,
            installed.Concat(new[] { eyeId, faceId, previousEyeId, previousFaceId }.OfType<string>()).Distinct().ToArray(),
            eyeId, faceId, previousEyeId, previousFaceId);

    public void Register(ModuleRelease release)
    {
        lock (_gate)
        {
            EnsureWritable();
            if (ReadRelease(release.Id) != release) throw new InvalidDataException("The module does not belong to this library.");
            var next = SelectionState(EyeId, FaceId, PreviousEyeId, PreviousFaceId, InstalledIds().Append(release.Id));
            WriteAtomic(_statePath, next); _state = next;
        }
    }

    /// <summary>Babble's two transports share a DLL, but keep their mutable configuration in separate profiles.</summary>
    public ModuleRelease CreateCompanion(ModuleRelease release)
    {
        lock (_gate)
        {
            EnsureWritable();
            if (ReadRelease(release.Id) != release || release.ModuleId != BaballoniaModuleId || !release.SupportsHeadset || !release.IsAvailable)
                throw new InvalidDataException("This package does not support both Babble inputs.");
            var id = Guid.NewGuid().ToString("N");
            var source = Path.Combine(_releasesRoot, release.Id, "payload");
            var target = Path.Combine(_releasesRoot, id, "payload");
            try
            {
                foreach (var file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories))
                {
                    var destination = GetContainedPath(target, Path.GetRelativePath(source, file));
                    Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                    File.Copy(file, destination);
                }
                var kind = release.Kind == ModuleKind.BabbleHeadsetRuntime ? ModuleKind.BaballoniaDesktop : ModuleKind.BabbleHeadsetRuntime;
                var companion = release with { Id = id, Kind = kind, DirectoryPath = Path.Combine(target, Path.GetRelativePath(source, release.DirectoryPath)) };
                WriteAtomic(Path.Combine(_releasesRoot, id, "release.json"), companion, false);
                return companion;
            }
            catch { if (Directory.Exists(target)) Directory.Delete(Path.GetDirectoryName(target)!, true); throw; }
        }
    }

    public void MigrateLegacyProfiles(TrackingInputSource source)
    {
        lock (_gate)
        {
            if (_state.Schema != 1 || IsReadOnly) return;
            var ids = InstalledIds();
            var incomplete = false;
            foreach (var id in ids)
            {
                var release = ReadRelease(id)!;
                // Old libraries can contain community/other headset modules.
                // Reclassifying those as Baballonia would corrupt their identity.
                if (release.ModuleId != BaballoniaModuleId || release.DllFileName != BaballoniaDll) continue;
                var supports = false;
                if (release.IsAvailable)
                    try { supports = HasHeadsetSupport(Path.Combine(release.DirectoryPath, release.DllFileName)); }
                    catch (Exception error) when (IsRecoveryError(error))
                    { AddRecoveryIssue("Legacy module " + id + " could not be inspected; its files were preserved: " + error.Message); }
                var migrated = release with { Kind = source == TrackingInputSource.BaballoniaDesktop ? ModuleKind.BaballoniaDesktop : ModuleKind.BabbleHeadsetRuntime, SupportsHeadset = supports };
                if (migrated == release) continue;
                try { EnsureWritable(); WriteAtomic(Path.Combine(_releasesRoot, id, "release.json"), migrated); }
                catch (Exception error) when (IsRecoveryError(error))
                {
                    incomplete = true;
                    // Do not auto-start a selected source using its old transport
                    // when the source-mode conversion could not be committed.
                    _state = _state with { Current = _state.Current == id ? null : _state.Current,
                        Previous = _state.Previous == id ? null : _state.Previous, Installed = ids };
                    AddRecoveryIssue("Legacy module " + id + " could not be migrated yet; select it again after its files are writable: " + error.Message);
                }
            }
            if (incomplete) return;
            var next = SelectionState(EyeId, FaceId, PreviousEyeId, PreviousFaceId, ids);
            try
            {
                EnsureWritable(); WriteAtomic(_statePath, next); _state = next;
                if (Current is { SupportsHeadset: true, IsAvailable: true } current)
                    Register(CreateCompanion(current));
            }
            catch (Exception error) when (IsRecoveryError(error))
            { AddRecoveryIssue("The legacy module library could not finish migrating yet. Existing profiles remain available: " + error.Message); }
        }
    }

    /// <summary>Stages a DLL or ZIP without loading module code or changing the selected module.</summary>
    public ModuleRelease Prepare(string path, ModuleKind? requestedKind = null)
    {
        EnsureWritable();
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (requestedKind is { } requested) _ = ModuleCatalog.Get(requested);
        var bareDll = string.Equals(Path.GetExtension(path), ".dll", StringComparison.OrdinalIgnoreCase);
        if (!bareDll && !string.Equals(Path.GetExtension(path), ".zip", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("Choose a compiled module DLL or a module ZIP.");
        using var source = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (source.Length is <= 0 or > MaxArchiveBytes)
            throw new InvalidDataException("The module file must be between 1 byte and 256 MB.");
        var hash = Convert.ToHexString(SHA256.HashData(source)).ToLowerInvariant();
        source.Position = 0;
        var id = Guid.NewGuid().ToString("N");
        var releaseRoot = Path.Combine(_releasesRoot, id);
        var staging = Path.Combine(_storeRoot, ".staging-" + id);
        var payload = Path.Combine(staging, "payload");
        Directory.CreateDirectory(payload);
        try
        {
            ModuleManifest? manifest = null;
            string? primaryPath = null;
            if (bareDll)
            {
                var filename = ValidateDllFileName(Path.GetFileName(path));
                primaryPath = GetContainedPath(payload, filename);
                using var output = new FileStream(primaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                CopyBounded(source, output, source.Length);
            }
            else
            {
                using var archive = new ZipArchive(source, ZipArchiveMode.Read, leaveOpen: true);
                if (archive.Entries.Count is 0 or > MaxEntries)
                    throw new InvalidDataException("The module ZIP has no files or too many entries.");
                var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var files = new List<(ZipArchiveEntry Entry, string Relative)>();
                long expanded = 0;
                foreach (var entry in archive.Entries)
                {
                    var relative = ValidateArchivePath(entry);
                    if (!paths.Add(relative.TrimEnd('/')))
                        throw new InvalidDataException("The module ZIP contains duplicate paths.");
                    if (entry.Length > MaxExpandedBytes - expanded)
                        throw new InvalidDataException("The module ZIP expands beyond 512 MB.");
                    expanded += entry.Length;
                    if (!relative.EndsWith('/')) files.Add((entry, relative));
                }
                var manifests = files.Where(f => string.Equals(Path.GetFileName(f.Relative), "module.json", StringComparison.OrdinalIgnoreCase)).ToArray();
                if (manifests.Length > 1) throw new InvalidDataException("The module ZIP contains multiple module.json manifests.");
                if (manifests.Length == 1)
                {
                    if (manifests[0].Entry.Length > 1024 * 1024) throw new InvalidDataException("The module manifest is too large.");
                    try
                    {
                        using var input = manifests[0].Entry.Open();
                        manifest = JsonSerializer.Deserialize<ModuleManifest>(input, JsonOptions)
                            ?? throw new InvalidDataException("The module manifest is empty.");
                    }
                    catch (JsonException error) { throw new InvalidDataException("The module manifest is not valid JSON.", error); }
                    var declared = ValidateManifest(manifest);
                    var manifestDirectory = Path.GetDirectoryName(manifests[0].Relative.Replace('/', Path.DirectorySeparatorChar)) ?? "";
                    var expected = Path.Combine(manifestDirectory, manifest.DllFileName!).Replace('\\', '/');
                    var matching = files.FirstOrDefault(f => string.Equals(f.Relative, expected, StringComparison.OrdinalIgnoreCase));
                    if (matching.Entry == null)
                        throw new InvalidDataException(declared.Kind == ModuleKind.VirtualDesktop
                            ? "The Virtual Desktop module DLL is missing. Download the compiled module package, not GitHub's Source code ZIP: https://download.vrdesktop.net/files/VirtualDesktop.VRCFaceTracking.zip"
                            : "The entry DLL named in module.json is missing beside the manifest. Import the compiled package or choose its module DLL directly.");
                    primaryPath = GetContainedPath(payload, matching.Relative);
                }
                foreach (var file in files)
                {
                    var destination = GetContainedPath(payload, file.Relative);
                    Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                    using var input = file.Entry.Open();
                    using var output = new FileStream(destination, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                    CopyBounded(input, output, file.Entry.Length);
                }
            }
            primaryPath ??= SelectEntryDll(payload);
            var identity = ReadAssemblyIdentity(primaryPath);
            var knownAssembly = ModuleCatalog.All.FirstOrDefault(x => x.AssemblyName == identity.Name);
            ModuleDefinition definition;
            if (manifest != null)
            {
                definition = ValidateManifest(manifest);
                if (definition.Kind != ModuleKind.Custom && definition.AssemblyName != identity.Name
                    || knownAssembly != null && knownAssembly.ModuleId != definition.ModuleId)
                    throw new InvalidDataException("The DLL does not match the module's declared identity.");
            }
            else if (knownAssembly != null)
            {
                definition = knownAssembly;
                var canonical = Path.Combine(Path.GetDirectoryName(primaryPath)!, definition.DllFileName);
                if (!string.Equals(primaryPath, canonical, StringComparison.OrdinalIgnoreCase))
                {
                    if (File.Exists(canonical)) throw new InvalidDataException("The ZIP has conflicting entry DLL names. Select the intended module DLL directly.");
                    File.Move(primaryPath, canonical);
                    primaryPath = canonical;
                }
            }
            else
            {
                var moduleId = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("BabbleBridge.Module|" + identity.StableIdentity)))[..32];
                definition = ModuleCatalog.Identify(Guid.ParseExact(moduleId, "N").ToString("D"), Path.GetFileName(primaryPath));
            }
            var supports = definition.ModuleId == BaballoniaModuleId && identity.Name == "VRCFaceTracking.Baballonia" && HasHeadsetSupport(primaryPath);
            var kind = requestedKind ?? (definition.Kind == ModuleKind.BabbleHeadsetRuntime && !supports ? ModuleKind.BaballoniaDesktop : definition.Kind);
            if (requestedKind is { } selected && (selected == ModuleKind.Custom
                    ? definition.Kind != ModuleKind.Custom : ModuleCatalog.Get(selected).ModuleId != definition.ModuleId))
                throw new InvalidDataException("The package does not match the selected module.");
            if (kind == ModuleKind.BabbleHeadsetRuntime && !supports)
                throw new InvalidDataException("This Baballonia package supports desktop input only. Import a package with Babble Frame Runtime support.");
            var version = manifest?.Version ?? identity.Version;
            var name = manifest?.ModuleName ?? (kind == ModuleKind.Custom ? identity.Name : definition.Name);
            var moduleDirectory = Path.GetDirectoryName(primaryPath)!;
            var release = new ModuleRelease(id, version, definition.ModuleId, Path.GetFileName(primaryPath), DateTimeOffset.UtcNow,
                hash, GetContainedPath(Path.Combine(releaseRoot, "payload"), Path.GetRelativePath(payload, moduleDirectory)), name, kind, supports, identity.Name);
            WriteAtomic(Path.Combine(staging, "release.json"), release, keepBackup: false);
            Directory.Move(staging, releaseRoot);
            return release;
        }
        catch
        {
            if (Directory.Exists(staging)) Directory.Delete(staging, recursive: true);
            throw;
        }
    }

    /// <summary>Atomically assigns a prepared release to both roles, preserving the previous role pair.</summary>
    public void Activate(ModuleRelease release)
    {
        ArgumentNullException.ThrowIfNull(release);
        lock (_gate)
        {
            EnsureWritable();
            var stored = ReadRelease(release.Id) ?? throw new InvalidDataException("The prepared module no longer exists.");
            if (stored != release) throw new InvalidDataException("The prepared module does not belong to this store.");
            ValidateAssembly(Path.Combine(stored.DirectoryPath, stored.DllFileName), stored.Kind == ModuleKind.Custom ? stored.AssemblyName! : ModuleCatalog.Get(stored.Kind).AssemblyName);
            if (EyeId == stored.Id && FaceId == stored.Id) return;
            var next = SelectionState(stored.Id, stored.Id, EyeId, FaceId, InstalledIds().Append(stored.Id));
            WriteAtomic(_statePath, next);
            _state = next;
        }
    }

    /// <summary>
    /// Selects registered releases for the eye and face roles in one transaction.
    /// A shared release is validated and instantiated only once by consumers.
    /// </summary>
    public void SelectChannels(ModuleRelease eye, ModuleRelease face)
    {
        ArgumentNullException.ThrowIfNull(eye);
        ArgumentNullException.ThrowIfNull(face);
        lock (_gate)
        {
            EnsureWritable();
            ValidateRegisteredSelection(eye, face);
            if (EyeId == eye.Id && FaceId == face.Id) return;
            var next = SelectionState(eye.Id, face.Id, EyeId, FaceId, InstalledIds());
            WriteAtomic(_statePath, next);
            _state = next;
        }
    }

    private void ValidateRegisteredSelection(ModuleRelease eye, ModuleRelease face)
    {
        var registered = InstalledIds().ToHashSet(StringComparer.Ordinal);
        foreach (var release in new[] { eye, face }.DistinctBy(module => module.Id))
        {
            if (!registered.Contains(release.Id) || ReadRelease(release.Id) != release)
                throw new InvalidDataException("The selected module does not belong to this library.");
            ValidateAssembly(Path.Combine(release.DirectoryPath, release.DllFileName),
                release.Kind == ModuleKind.Custom ? release.AssemblyName! : ModuleCatalog.Get(release.Kind).AssemblyName);
        }
        // Do not let two values with the same ID bypass identity validation.
        if (eye.Id == face.Id && eye != face)
            throw new InvalidDataException("The selected module does not belong to this library.");
    }

    public ModuleRelease Install(string zipPath)
    {
        var release = Prepare(zipPath);
        Activate(release);
        return release;
    }

    public ModuleRelease Rollback()
        => RestorePreviousChannels().PrimaryModule!;

    /// <summary>Restores the whole previous role pair; invoking again undoes the restore.</summary>
    public ModuleChannelSelection RestorePreviousChannels()
    {
        lock (_gate)
        {
            EnsureWritable();
            var selection = ReadSelection(PreviousEyeId, PreviousFaceId);
            if (!selection.IsComplete) throw new InvalidOperationException("There is no previous module selection to restore.");
            ValidateRegisteredSelection(selection.EyeModule!, selection.FaceModule!);
            var next = SelectionState(PreviousEyeId, PreviousFaceId, EyeId, FaceId, InstalledIds());
            WriteAtomic(_statePath, next);
            _state = next;
            return selection;
        }
    }

    /// <summary>Removes one imported profile. The caller must stop its worker before removing an active profile.</summary>
    public void Remove(ModuleRelease release)
    {
        ArgumentNullException.ThrowIfNull(release);
        lock (_gate)
        {
            EnsureWritable();
            var stored = ReadRelease(release.Id);
            if (stored != release || !InstalledIds().Contains(release.Id, StringComparer.Ordinal))
                throw new InvalidDataException("The module does not belong to this library.");
            var releaseRoot = Path.Combine(_releasesRoot, stored.Id);
            var removedRoot = Path.Combine(_storeRoot, ".removed-" + stored.Id);
            EnsureRemovalTree(releaseRoot);
            EnsureNoReparsePoint(_statePath);
            EnsureNoReparsePoint(_statePath + ".backup");
            if (Directory.Exists(removedRoot) || File.Exists(removedRoot))
                throw new IOException("A previous removal is still being cleaned up. Reopen the app and try again.");
            // Losing either role stops the pair. Never silently repurpose the
            // surviving source as the missing role, including during recovery.
            var selected = EyeId == stored.Id || FaceId == stored.Id;
            var previous = PreviousEyeId == stored.Id || PreviousFaceId == stored.Id;
            var next = SelectionState(selected ? null : EyeId, selected ? null : FaceId,
                previous ? null : PreviousEyeId, previous ? null : PreviousFaceId,
                InstalledIds().Where(id => id != stored.Id));
            var backupPath = _statePath + ".backup";
            var oldBackup = File.Exists(backupPath) ? File.ReadAllBytes(backupPath) : null;
            var moved = false;
            try
            {
                // Prepare safe recovery before moving any files: if the process stops between
                // the move and main-state commit, this backup already excludes the removed ID.
                WriteAtomic(backupPath, next, keepBackup: false);
                MoveRemovalDirectory(releaseRoot, removedRoot);
                moved = true;
                WriteAtomic(_statePath, next, keepBackup: false);
                _state = next;
            }
            catch
            {
                if (moved)
                {
                    try { MoveRemovalDirectory(removedRoot, releaseRoot); }
                    catch (Exception restoreError) when (restoreError is IOException or UnauthorizedAccessException)
                    {
                        // Keep the sanitized backup valid if a concurrent file lock prevents
                        // rollback. Reopening can finish recovery without losing another profile.
                        _state = next;
                        throw new IOException("Module removal was interrupted. Reopen the app to finish recovering the module library.", restoreError);
                    }
                }
                try
                {
                    if (oldBackup != null) WriteAtomicContent(backupPath, stream => stream.Write(oldBackup), false);
                    else if (File.Exists(backupPath)) File.Delete(backupPath);
                }
                catch (Exception restoreError) when (restoreError is IOException or UnauthorizedAccessException)
                { /* The unchanged primary and restored payload still describe the old library. */ }
                throw;
            }
            TryDeleteRemovedRelease(removedRoot, stored.Id);
        }
    }

    private void MoveRemovalDirectory(string source, string destination)
    {
        // Windows scanners can briefly open a just-inspected module directory
        // without delete sharing. Retry only the owned atomic rename, keeping
        // the transaction uncommitted and preserving real permission failures.
        ReadOnlySpan<int> delays = [20, 40, 70, 110];
        for (var attempt = 0; ; attempt++)
        {
            if (attempt > 0) EnsureRemovalTree(source);
            try { Directory.Move(source, destination); return; }
            catch (Exception error) when (OperatingSystem.IsWindows() && attempt < delays.Length
                && (error is IOException or UnauthorizedAccessException)
                && ((error.HResult & 0xffff) is 5 or 32 or 33))
            {
                Thread.Sleep(delays[attempt]);
            }
        }
    }

    private void CleanupRemovedReleases()
    {
        EnsureNoReparsePoint(_storeRoot);
        var installed = InstalledIds();
        foreach (var directory in Directory.EnumerateDirectories(_storeRoot, ".removed-*", SearchOption.TopDirectoryOnly))
        {
            var id = Path.GetFileName(directory)[".removed-".Length..];
            if (Guid.TryParseExact(id, "N", out _) && !installed.Contains(id, StringComparer.Ordinal)
                && !Directory.Exists(Path.Combine(_releasesRoot, id)))
                TryDeleteRemovedRelease(directory, id);
        }
    }

    private void TryDeleteRemovedRelease(string directory, string id)
    {
        try
        {
            EnsureRemovalTree(directory);
            var metadata = JsonSerializer.Deserialize<ModuleRelease>(File.ReadAllText(Path.Combine(directory, "release.json")), JsonOptions);
            if (metadata?.Id != id) return;
            var originalPayload = Path.Combine(_releasesRoot, id, "payload");
            _ = GetContainedPath(originalPayload, Path.GetRelativePath(originalPayload, metadata.DirectoryPath));
            DeleteRemovedTree(directory, Path.Combine(directory, "release.json"));
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidDataException or JsonException or ArgumentException)
        { /* Locked payloads stay quarantined and are retried when the library is reopened. */ }
    }

    private static void DeleteRemovedTree(string directory, string metadataPath)
    {
        foreach (var entry in Directory.EnumerateFileSystemEntries(directory))
        {
            if (string.Equals(entry, metadataPath, StringComparison.OrdinalIgnoreCase)) continue;
            EnsureNoReparsePoint(entry);
            if ((File.GetAttributes(entry) & FileAttributes.Directory) != 0) DeleteRemovedTree(entry, metadataPath);
            else File.Delete(entry);
        }
        // Retain ownership metadata until every payload entry is gone, allowing
        // a partially deleted, locked directory to be cleaned safely on reopen.
        if (File.Exists(metadataPath) && Path.GetDirectoryName(metadataPath) == directory) File.Delete(metadataPath);
        Directory.Delete(directory);
    }

    private void EnsureRemovalTree(string directory)
    {
        var root = Path.GetFullPath(directory);
        _ = GetContainedPath(_storeRoot, Path.GetRelativePath(_storeRoot, root));
        EnsureNoReparsePoint(Path.GetDirectoryName(_storeRoot)!);
        EnsureNoReparsePoint(_storeRoot);
        EnsureNoReparsePoint(_releasesRoot);
        EnsureNoReparsePoint(root);
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.TryPop(out var parent))
        {
            foreach (var entry in Directory.EnumerateFileSystemEntries(parent))
            {
                var attributes = File.GetAttributes(entry);
                if ((attributes & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidDataException("Module removal cannot follow symbolic links or junctions.");
                if ((attributes & FileAttributes.Directory) != 0) pending.Push(entry);
            }
        }
    }

    private static void EnsureNoReparsePoint(string path)
    {
        FileAttributes attributes;
        try { attributes = File.GetAttributes(path); }
        catch (Exception error) when (error is FileNotFoundException or DirectoryNotFoundException) { return; }
        if ((attributes & FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("Module removal cannot follow symbolic links or junctions.");
    }

    private StoreState ReadState()
    {
        if (!File.Exists(_statePath) && !File.Exists(_statePath + ".backup")) return new StoreState(1, null, null);
        // Notice future schemas before selecting a fallback. Even a readable
        // older backup must not allow this app to overwrite a newer manifest.
        foreach (var candidate in new[] { _statePath, _statePath + ".backup" })
            if (File.Exists(candidate))
                try
                {
                    if (new FileInfo(candidate).Length > MaximumPreservedStateBytes)
                    {
                        MarkOversizedStateReadOnly(candidate);
                        continue;
                    }
                    using var document = JsonDocument.Parse(ReadBoundedMetadata(candidate));
                    if (document.RootElement.ValueKind == JsonValueKind.Object)
                        foreach (var property in document.RootElement.EnumerateObject())
                            if (property.Name.Equals("Schema", StringComparison.OrdinalIgnoreCase)
                                && property.Value.ValueKind == JsonValueKind.Number && property.Value.TryGetInt32(out var version)
                                && version is not 1 and not 2 and not 3)
                            {
                                _readOnlyReason = "This module library was saved by another app version. Modules can be reviewed, but changes are disabled to preserve it.";
                                AddRecoveryIssue("Unsupported saved module schema " + version + ": " + candidate);
                            }
                }
                catch (Exception error) when (IsRecoveryError(error)) { }
        foreach (var path in new[] { _statePath, _statePath + ".backup" })
        {
            if (!File.Exists(path)) continue;
            try
            {
                var state = JsonSerializer.Deserialize<StoreState>(ReadBoundedMetadata(path), JsonOptions)
                    ?? throw new InvalidDataException("Empty module state.");
                if (state.Schema is not 1 and not 2 and not 3) throw new InvalidDataException("Unsupported module state version.");
                if (state.Installed?.Length > MaximumSavedModules) throw new InvalidDataException("The saved module list is too large.");
                foreach (var id in new[] { state.Current, state.Previous, state.EyeModuleId, state.FaceModuleId, state.PreviousEyeId, state.PreviousFaceId })
                    if (id is not null && !Guid.TryParseExact(id, "N", out _)) throw new InvalidDataException("Invalid selected module ID.");
                if (state.Schema == 3)
                {
                    if ((state.EyeModuleId is null) != (state.FaceModuleId is null)
                        || (state.PreviousEyeId is null) != (state.PreviousFaceId is null)
                        || state.Current != state.EyeModuleId || state.Previous != state.PreviousEyeId)
                        throw new InvalidDataException("The saved tracking role selection is inconsistent.");
                }
                if (ReferencesPendingRemoval(state)) throw new InvalidDataException("An interrupted module removal requires its sanitized backup.");
                var sanitized = QuarantineInvalidEntries(state);
                // Repair the primary after recovery, so a later normal save cannot replace
                // the valid backup with a corrupt or interrupted-removal manifest.
                if (path != _statePath && !IsReadOnly)
                {
                    try { PreserveStateFiles(); WriteAtomic(_statePath, state, keepBackup: false); }
                    catch (Exception error) when (IsRecoveryError(error))
                    { AddRecoveryIssue("The saved backup is usable, but its primary state could not be repaired yet: " + error.Message); }
                }
                return sanitized;
            }
            catch (Exception error) when (IsRecoveryError(error))
            { _preserveStateOnNextWrite = true; AddRecoveryIssue("Saved module state could not be read: " + path + ": " + error.Message); }
        }
        // Both manifests may have been interrupted. Recover only independently
        // validated profile metadata; never guess which source should start.
        var recovered = new List<string>();
        if (Directory.Exists(_releasesRoot))
            try
            {
                foreach (var directory in Directory.EnumerateDirectories(_releasesRoot).Take(MaximumSavedModules))
                {
                    var id = Path.GetFileName(directory);
                    if (!Guid.TryParseExact(id, "N", out _)) continue;
                    try { EnsureNoReparsePoint(directory); if (ReadRelease(id) is not null) recovered.Add(id); }
                    catch (Exception error) when (IsRecoveryError(error)) { AddRecoveryIssue("Module profile " + id + " needs repair: " + error.Message); }
                }
            }
            catch (Exception error) when (IsRecoveryError(error)) { AddRecoveryIssue("Module profiles could not all be listed: " + error.Message); }
        return SelectionState(null, null, null, null, recovered);
    }

    private StoreState QuarantineInvalidEntries(StoreState state)
    {
        var raw = (state.Installed ?? []).Concat(new[] { state.Current, state.Previous, state.EyeModuleId,
            state.FaceModuleId, state.PreviousEyeId, state.PreviousFaceId }.OfType<string>())
            .Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        var valid = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var id in raw)
            try
            {
                if (id is null || ReadRelease(id) is null) throw new InvalidDataException("An installed module ID is missing.");
                valid.Add(id);
            }
            catch (Exception error) when (IsRecoveryError(error))
            {
                _preserveStateOnNextWrite = true;
                AddRecoveryIssue("Module profile " + (id ?? "(missing ID)") + " was kept out of the active library; its files were preserved: " + error.Message);
            }
        bool Available(string? id) => id is not null && valid.Contains(id);
        if (state.Schema < 3)
            return state with { Current = Available(state.Current) ? state.Current : null,
                Previous = Available(state.Previous) ? state.Previous : null, Installed = valid.ToArray() };
        var selected = Available(state.EyeModuleId) && Available(state.FaceModuleId);
        var previous = Available(state.PreviousEyeId) && Available(state.PreviousFaceId);
        return state with
        {
            Current = selected ? state.Current : null, EyeModuleId = selected ? state.EyeModuleId : null, FaceModuleId = selected ? state.FaceModuleId : null,
            Previous = previous ? state.Previous : null, PreviousEyeId = previous ? state.PreviousEyeId : null, PreviousFaceId = previous ? state.PreviousFaceId : null,
            Installed = valid.ToArray()
        };
    }

    private bool ReferencesPendingRemoval(StoreState state)
    {
        foreach (var id in (state.Installed ?? []).Concat(new[] { state.Current, state.Previous, state.EyeModuleId,
            state.FaceModuleId, state.PreviousEyeId, state.PreviousFaceId }.OfType<string>()).Distinct())
        {
            if (!Guid.TryParseExact(id, "N", out _) || Directory.Exists(Path.Combine(_releasesRoot, id))) continue;
            var removed = Path.Combine(_storeRoot, ".removed-" + id);
            if (!Directory.Exists(removed)) continue;
            try
            {
                EnsureNoReparsePoint(removed);
                var metadata = JsonSerializer.Deserialize<ModuleRelease>(ReadBoundedMetadata(Path.Combine(removed, "release.json")), JsonOptions);
                if (metadata?.Id == id) return true;
            }
            catch (Exception error) when (IsRecoveryError(error)) { }
        }
        return false;
    }

    private static byte[] ReadBoundedMetadata(string path)
    {
        using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (input.Length is <= 0 or > MaximumStateBytes) throw new InvalidDataException("The saved module metadata is empty or too large.");
        var bytes = new byte[(int)input.Length]; input.ReadExactly(bytes); return bytes;
    }

    private void AddRecoveryIssue(string message)
    {
        if (_recoveryIssues.Count < 128 && !_recoveryIssues.Contains(message, StringComparer.Ordinal)) _recoveryIssues.Add(message);
    }

    private static bool IsRecoveryError(Exception error) => error is IOException or UnauthorizedAccessException or InvalidDataException
        or JsonException or ArgumentException or BadImageFormatException or NotSupportedException or InvalidOperationException;

    private void PreserveStateFiles()
    {
        EnsureNoReparsePoint(Path.GetDirectoryName(_storeRoot)!); EnsureNoReparsePoint(_storeRoot);
        foreach (var path in new[] { _statePath, _statePath + ".backup" })
        {
            EnsureNoReparsePoint(path);
            if (!File.Exists(path)) continue;
            using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (input.Length > MaximumPreservedStateBytes)
            {
                MarkOversizedStateReadOnly(path);
                throw new InvalidOperationException(_readOnlyReason);
            }
            var preserved = path + ".recovery-" + Guid.NewGuid().ToString("N");
            // Preservation is a raw bounded copy, deliberately independent of
            // JSON parsing. Empty/truncated/oversized-for-parsing files are the
            // exact inputs that must remain recoverable before a repair.
            try
            {
                using var output = new FileStream(preserved, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                    65536, FileOptions.WriteThrough);
                var buffer = new byte[65536]; long copied = 0;
                int count;
                while ((count = input.Read(buffer)) > 0)
                {
                    if (count > MaximumPreservedStateBytes - copied)
                    {
                        MarkOversizedStateReadOnly(path);
                        throw new InvalidOperationException(_readOnlyReason);
                    }
                    output.Write(buffer, 0, count); copied += count;
                }
                output.Flush(flushToDisk: true);
            }
            catch
            {
                try { File.Delete(preserved); } catch (Exception error) when (IsRecoveryError(error)) { }
                throw;
            }
        }
        _preserveStateOnNextWrite = false;
    }

    private void MarkOversizedStateReadOnly(string path)
    {
        _readOnlyReason = "A saved module manifest is too large to preserve automatically. The library is open read-only; existing files have been left untouched.";
        AddRecoveryIssue("Saved module state exceeds the 32 MB recovery archive limit: " + path);
    }

    private void EnsureWritable()
    {
        if (_readOnlyReason is not null) throw new InvalidOperationException(_readOnlyReason);
        if (_preserveStateOnNextWrite) PreserveStateFiles();
    }

    private ModuleRelease? ReadRelease(string? id)
    {
        if (id is null) return null;
        if (!Guid.TryParseExact(id, "N", out _)) throw new InvalidDataException("Invalid saved release ID.");
        var releaseRoot = Path.Combine(_releasesRoot, id);
        var release = JsonSerializer.Deserialize<ModuleRelease>(ReadBoundedMetadata(Path.Combine(releaseRoot, "release.json")), JsonOptions)
            ?? throw new InvalidDataException("Empty saved release.");
        var payloadRoot = Path.Combine(releaseRoot, "payload");
        var directory = Path.GetFullPath(release.DirectoryPath);
        var definition = ModuleCatalog.Get(release.Kind);
        var custom = release.Kind == ModuleKind.Custom;
        _ = ValidateDllFileName(release.DllFileName);
        if (string.IsNullOrWhiteSpace(release.Version) || release.Version.Length > 80 || release.Version.Any(char.IsControl)
            || string.IsNullOrWhiteSpace(release.ModuleName) || release.ModuleName.Length > 200 || release.ModuleName.Any(char.IsControl)
            || release.Sha256 is not { Length: 64 } || release.Sha256.Any(value => !Uri.IsHexDigit(value)))
            throw new InvalidDataException("The saved module display or checksum metadata is invalid.");
        if (!string.Equals(release.Id, id, StringComparison.OrdinalIgnoreCase) || !Guid.TryParse(release.ModuleId, out _)
            || !custom && (release.ModuleId != definition.ModuleId || release.DllFileName != definition.DllFileName
                || release.AssemblyName != null && release.AssemblyName != definition.AssemblyName)
            || custom && release.AssemblyName is not null && (string.IsNullOrWhiteSpace(release.AssemblyName) || release.AssemblyName.Length > 200 || release.AssemblyName.Any(char.IsControl))
            ||
            !(directory.Equals(payloadRoot, StringComparison.OrdinalIgnoreCase) || directory.StartsWith(payloadRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidDataException("The saved module release is invalid or missing.");
        if (custom)
        {
            // These providers were once ordinary custom DLL imports. A later
            // catalog must not make that previously valid profile poison the
            // entire library. Confirm the original metadata-derived identity;
            // never load module code or trust a claimed catalog name alone.
            var promoted = PromoteLegacyCatalogProfile(release);
            if (promoted is not null) return promoted;
            if (string.IsNullOrWhiteSpace(release.AssemblyName))
                throw new InvalidDataException("The saved custom assembly identity is missing.");
            if (ModuleCatalog.Identify(release.ModuleId, release.DllFileName).Kind != ModuleKind.Custom
                || ModuleCatalog.All.Any(x => x.AssemblyName == release.AssemblyName))
                throw new InvalidDataException("The saved custom module conflicts with a known module identity.");
            // Library recovery must not depend on an intact payload. A corrupt
            // imported DLL remains visible/removable, just like a missing one;
            // activation and role selection validate its binary before commit.
        }
        return release;
    }

    private static ModuleRelease? PromoteLegacyCatalogProfile(ModuleRelease release)
    {
        var known = ModuleCatalog.All.FirstOrDefault(definition => definition.Kind is ModuleKind.EyeTrackVR or ModuleKind.Beyond or ModuleKind.SRanipal
            && string.Equals(definition.DllFileName, release.DllFileName, StringComparison.OrdinalIgnoreCase));
        if (known is null) return null;
        var identity = ReadAssemblyIdentity(Path.Combine(release.DirectoryPath, release.DllFileName));
        if (identity.Name != known.AssemblyName || !identity.TrackingModule
            || release.AssemblyName is not null && release.AssemblyName != identity.Name)
            throw new InvalidDataException("The older custom module does not match its known entry assembly.");
        var legacyId = Guid.ParseExact(Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("BabbleBridge.Module|" + identity.StableIdentity)))[..32], "N");
        var savedId = Guid.Parse(release.ModuleId);
        if (savedId != legacyId && savedId != Guid.Parse(known.ModuleId))
            throw new InvalidDataException("The older custom module ID does not match its original assembly identity.");
        return release with { Kind = known.Kind, ModuleId = known.ModuleId, DllFileName = known.DllFileName, AssemblyName = known.AssemblyName };
    }

    private static string ValidateArchivePath(ZipArchiveEntry entry)
    {
        var name = entry.FullName.Replace('\\', '/');
        if (string.IsNullOrWhiteSpace(name) || name.StartsWith('/') || name.Contains(':') || name.Length > 220)
            throw new InvalidDataException("The module ZIP contains an unsafe path.");
        var unixType = (entry.ExternalAttributes >> 16) & 0xF000;
        if (unixType is not 0 and not 0x8000 and not 0x4000 || (entry.ExternalAttributes & (int)FileAttributes.ReparsePoint) != 0)
            throw new InvalidDataException("Symbolic links and special files are not supported in module ZIPs.");
        var components = name.TrimEnd('/').Split('/');
        foreach (var part in components)
        {
            if (part is "" or "." or ".." || part.EndsWith(' ') || part.EndsWith('.') || part.IndexOfAny(['<', '>', '"', '|', '?', '*', '\0']) >= 0)
                throw new InvalidDataException("The module ZIP contains an unsafe path.");
            var stem = part.Split('.')[0].ToUpperInvariant();
            if (stem is "CON" or "PRN" or "AUX" or "NUL" ||
                (stem.Length == 4 && (stem.StartsWith("COM") || stem.StartsWith("LPT")) && stem[3] is >= '0' and <= '9'))
                throw new InvalidDataException("The module ZIP contains a reserved Windows filename.");
        }
        return name;
    }

    private static ModuleDefinition ValidateManifest(ModuleManifest manifest)
    {
        _ = ValidateDllFileName(manifest.DllFileName);
        var definition = ModuleCatalog.Identify(manifest.ModuleId, manifest.DllFileName);
        if (manifest.Version != null && (string.IsNullOrWhiteSpace(manifest.Version) || manifest.Version.Length > 80 || manifest.Version.Any(char.IsControl)))
            throw new InvalidDataException("The module manifest must specify a valid version.");
        if (manifest.ModuleName?.Length > 200 || manifest.ModuleName?.Any(char.IsControl) == true)
            throw new InvalidDataException("The module name is invalid.");
        return definition;
    }

    private static void ValidateAssembly(string path, string expectedName)
    {
        if (ReadAssemblyIdentity(path).Name != expectedName)
            throw new InvalidDataException("The DLL does not match the module's declared entry assembly.");
    }

    private static string ValidateDllFileName(string? name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Length > 200 || name.IndexOfAny(['/', '\\', ':', '<', '>', '"', '|', '?', '*']) >= 0
            || name.Any(char.IsControl) || name.EndsWith(' ') || name.EndsWith('.') || !name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("The entry DLL must be a safe DLL filename without a folder path.");
        var stem = name.Split('.')[0].ToUpperInvariant();
        if (stem is "CON" or "PRN" or "AUX" or "NUL" || stem.Length == 4 && (stem.StartsWith("COM") || stem.StartsWith("LPT")) && stem[3] is >= '0' and <= '9')
            throw new InvalidDataException("The entry DLL uses a reserved Windows filename.");
        return name;
    }

    private static string SelectEntryDll(string payload)
    {
        var assemblies = new List<(string Path, AssemblyIdentity Identity)>();
        foreach (var path in Directory.EnumerateFiles(payload, "*", SearchOption.AllDirectories)
            .Where(path => path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)))
        {
            try { assemblies.Add((path, ReadAssemblyIdentity(path))); }
            catch (InvalidDataException) { /* Native dependencies are allowed beside the managed entry DLL. */ }
        }
        var modules = assemblies.Where(x => x.Identity.TrackingModule).ToArray();
        if (modules.Length == 1) return modules[0].Path;
        if (modules.Length == 0 && assemblies.Count == 1) return assemblies[0].Path;
        throw new InvalidDataException(assemblies.Count == 0
            ? "No managed module DLL was found. Download a compiled module package instead of source code."
            : "The ZIP has multiple possible module DLLs. Select the module DLL directly, or add module.json naming its entry DLL.");
    }

    private sealed record AssemblyIdentity(string Name, string Version, string StableIdentity, bool TrackingModule);
    private static AssemblyIdentity ReadAssemblyIdentity(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            using var pe = new PEReader(stream);
            if (!pe.HasMetadata) throw new InvalidDataException("The module DLL is not a managed .NET assembly.");
            var metadata = pe.GetMetadataReader();
            if (!metadata.IsAssembly) throw new InvalidDataException("The module DLL is not a managed assembly.");
            var assembly = metadata.GetAssemblyDefinition();
            var name = metadata.GetString(assembly.Name);
            if (string.IsNullOrWhiteSpace(name) || name.Length > 200 || name.Any(char.IsControl))
                throw new InvalidDataException("The module assembly name is invalid.");
            bool IsTrackingBase(EntityHandle handle, HashSet<EntityHandle> seen)
            {
                if (handle.IsNil || !seen.Add(handle)) return false;
                if (handle.Kind == HandleKind.TypeReference)
                {
                    var reference = metadata.GetTypeReference((TypeReferenceHandle)handle);
                    return metadata.GetString(reference.Name) == "ExtTrackingModule" && metadata.GetString(reference.Namespace) == "VRCFaceTracking";
                }
                return handle.Kind == HandleKind.TypeDefinition && IsTrackingBase(metadata.GetTypeDefinition((TypeDefinitionHandle)handle).BaseType, seen);
            }
            var tracking = metadata.TypeDefinitions.Any(handle =>
            {
                var type = metadata.GetTypeDefinition(handle);
                return (type.Attributes & TypeAttributes.Abstract) == 0 && IsTrackingBase(type.BaseType, []);
            });
            return new(name, assembly.Version.ToString(), name + "|" + Convert.ToHexString(metadata.GetBlobBytes(assembly.PublicKey)), tracking);
        }
        catch (BadImageFormatException ex) { throw new InvalidDataException("The module DLL is not a valid .NET assembly.", ex); }
    }

    private static bool HasHeadsetSupport(string path)
    {
        using var stream = File.OpenRead(path);
        using var pe = new PEReader(stream);
        var metadata = pe.GetMetadataReader();
        return metadata.TypeDefinitions.Any(handle => metadata.GetString(metadata.GetTypeDefinition(handle).Name).Contains("BabbleLink", StringComparison.OrdinalIgnoreCase))
            || metadata.FieldDefinitions.Any(handle => metadata.GetString(metadata.GetFieldDefinition(handle).Name).Equals("babbleLink", StringComparison.OrdinalIgnoreCase));
    }

    private static string GetContainedPath(string root, string relative)
    {
        var full = Path.GetFullPath(Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar)));
        if (!full.Equals(root, StringComparison.OrdinalIgnoreCase) && !full.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("The module ZIP contains a path outside its release folder.");
        return full;
    }

    private static void CopyBounded(Stream input, Stream output, long expectedLength)
    {
        var buffer = new byte[81920];
        long total = 0;
        int read;
        while ((read = input.Read(buffer, 0, buffer.Length)) != 0)
        {
            total += read;
            if (total > expectedLength || total > MaxExpandedBytes)
                throw new InvalidDataException("A ZIP entry expanded beyond its advertised size.");
            output.Write(buffer, 0, read);
        }
        if (total != expectedLength) throw new InvalidDataException("A ZIP entry is incomplete.");
    }

    internal static void WriteAtomic<T>(string path, T value, bool keepBackup = true)
        => WriteAtomicContent(path, stream => JsonSerializer.Serialize(stream, value, JsonOptions), keepBackup);

    private static void WriteAtomicContent(string path, Action<Stream> write, bool keepBackup)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporary = path + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096, FileOptions.WriteThrough))
            {
                write(output);
                output.Flush(flushToDisk: true);
            }
            if (File.Exists(path) && keepBackup)
                File.Replace(temporary, path, path + ".backup", ignoreMetadataErrors: true);
            else File.Move(temporary, path, overwrite: true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    private sealed record StoreState(int Schema, string? Current, string? Previous, string[]? Installed = null,
        string? EyeModuleId = null, string? FaceModuleId = null, string? PreviousEyeId = null, string? PreviousFaceId = null);
    private sealed class ModuleManifest
    {
        public string? ModuleId { get; set; }
        public string? Version { get; set; }
        public string? DllFileName { get; set; }
        public string? ModuleName { get; set; }
    }
}
