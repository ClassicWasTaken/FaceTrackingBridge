// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0

using System.Windows;
using System.IO;

namespace BabbleFrameBridge.App;

public partial class MainWindow
{
    private void RefreshKittyMemorySummary()
    {
        if (KittyMemorySummary is not null)
            KittyMemorySummary.Text = KittyOverlay.MemorySummary;
    }

    private void ForgetKittyMemory_Click(object sender, RoutedEventArgs e)
    {
        if (_closing) return;
        try
        {
            KittyOverlay.ForgetMemory();
            RefreshKittyMemorySummary();
            ShowNotice("Relay's saved preferences and recent context have been forgotten.");
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidOperationException)
        {
            Log("Relay's memory could not be cleared: " + error.Message);
            RefreshKittyMemorySummary();
            ShowNotice("Relay's memory could not be cleared. Check access to the app data folder and try again.", true);
        }
    }
}
