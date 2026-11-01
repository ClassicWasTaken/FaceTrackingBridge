// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0

namespace BabbleFrameBridge.App;

public partial class MainWindow
{
    private void EnsureModuleLibraryWritable()
    {
        if (_store.IsReadOnly)
            throw new InvalidOperationException(_store.ReadOnlyReason ?? "This module library is read-only. Your original module files have been preserved.");
    }

    private void ShowModuleRecoveryNotice()
    {
        if (_store.RecoveryNotice is { Length: > 0 } notice)
        {
            Log(notice);
            ShowNotice(notice, true);
        }
    }
}
