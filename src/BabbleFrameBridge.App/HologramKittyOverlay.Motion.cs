// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0

namespace BabbleFrameBridge.App;

public sealed partial class HologramKittyOverlay
{
    private readonly KittyMotionController _motions = new();
    private KittyMotionPose _motionPose = KittyMotionPose.Idle;
    private double _motionLift;
    public KittyMotionSkill? PracticingSkill => _motions.CurrentSkill;
    public int LearnedMotionCount => _motions.KnownSkillCount;

    private bool HasMotionSpace => _worldMode == WorldMode.Free && !_carry.IsPressed
        && _position.X >= 24 && _position.X <= ActualWidth - 168 && _position.Y >= 50;

    private void AdvanceMotion(double elapsedSeconds)
    {
        var completed = _motions.CompletedPlanCount;
        var policy = _mind.BehaviorProbabilities;
        var playfulIntent = policy[(int)KittyAction.Roll] + policy[(int)KittyAction.Chase]
            + policy[(int)KittyAction.Zoomies] + policy[(int)KittyAction.Pounce]
            + policy[(int)KittyAction.TailChase];
        var drive = Math.Clamp(_mind.Willingness * .5 + Math.Clamp(playfulIntent * 2, 0, 1) * .5, 0, 1);
        _motionPose = _motions.Tick(elapsedSeconds, _mind.Action, HasMotionSpace,
            _mind.Energy, _mind.Bond, _mind.Playfulness,
            neuralDrive: drive);
        ApplyMotionStyle();
        _motionLift = _motionPose.Active ? _motionPose.Lift + _cat.MotionGroundLift : 0;
        _lift += _motionLift;
        PlaceCat();
        if (_motions.CompletedPlanCount != completed) MarkMemoryDirty();
    }

    private void ApplyMotionStyle()
    {
        var pose = _motionPose.Active ? _motionPose : KittyMotionPose.Idle;
        _cat.SetMotionStyle(pose.RollDegrees, pose.StretchX, pose.StretchY,
            1 + pose.Stride * .25, pose.Tail);
    }

    private void CancelMotion()
    {
        _motions.Cancel(); _motionPose = KittyMotionPose.Idle;
        _lift = Math.Max(0, _lift - _motionLift); _motionLift = 0;
        _cat.SetMotionStyle();
    }

    private void RestoreMotionLearning(KittyMemory memory) => _motions.RestoreLearning(
        memory.MotionPreferences, memory.MotionMastery, memory.MotionPrototypes,
        memory.MotionPractice, memory.MotionKnownMask);

    private KittyMemory WithMotionLearning(KittyMemory memory)
    {
        var learned = _motions.ExportLearning();
        return memory with { MotionPreferences = learned.Preferences, MotionMastery = learned.Mastery,
            MotionPrototypes = learned.Prototypes, MotionPractice = learned.Practice, MotionKnownMask = learned.KnownMask };
    }
}
