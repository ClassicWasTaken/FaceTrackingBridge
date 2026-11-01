// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0
using BabbleFrameBridge.App;

var checks = 0;
void Check(bool value, string message) { if (!value) throw new Exception(message); checks++; }
void PoseSafe(KittyMotionPose p)
{
    Check(new[] { p.Lift, p.RollDegrees, p.StretchX, p.StretchY, p.Stride, p.Tail, p.SpeedMultiplier, p.Phase }
        .All(double.IsFinite), "A procedural pose contains nonfinite output.");
    Check(p.Lift is >= 0 and <= 42 && Math.Abs(p.RollDegrees) <= 360
        && p.StretchX is >= .8 and <= 1.2 && p.StretchY is >= .8 and <= 1.2
        && Math.Abs(p.Stride) <= 1 && Math.Abs(p.Tail) <= 1 && p.SpeedMultiplier is >= .6 and <= 1.4,
        "A generated pose escaped its physical envelope.");
}
void Finish(KittyMotionController c)
{
    for (var i = 0; i < 120 && c.CurrentPlan is not null; i++)
        PoseSafe(c.Tick(.05, KittyAction.Watch, true));
    Check(c.CurrentPlan is null, "A bounded plan never completed.");
}

var fresh = new KittyMotionController(123);
Check(fresh.KnownSkillCount == 2 && fresh.IsKnown(KittyMotionSkill.Hop) && fresh.IsKnown(KittyMotionSkill.StretchWave),
    "A new kitty did not start with its two basic skills.");
Check(!fresh.TryPlan(KittyMotionSkill.Tumble), "An undiscovered skill could be forced into the first repertoire.");
for (var i = 0; i < 18; i++)
{
    Check(fresh.TryPlan(i % 2 == 0 ? KittyMotionSkill.Hop : KittyMotionSkill.StretchWave), "A basic practice plan did not start.");
    Finish(fresh);
}
Check(fresh.KnownSkillCount == 6 && fresh.CompletedPlanCount == 18,
    "Autonomous practice could not discover the finite complete repertoire without clicking.");
Check(fresh.Mastery(KittyMotionSkill.Hop) > .18, "Practice did not improve skill mastery.");
var learned = fresh.ExportLearning();
Check(learned.Practice.Sum() == 18, "Successful practice was not represented in exported learning.");

foreach (var skill in Enum.GetValues<KittyMotionSkill>())
{
    var signatures = new HashSet<KittyMotionPlan>();
    for (var trial = 0; trial < 6; trial++)
    {
        Check(fresh.TryPlan(skill), "A known skill could not generate a new plan.");
        signatures.Add(fresh.CurrentPlan!.Value);
        var sawLiftAndStride = false; var sawRoll = false;
        for (var frame = 0; frame < 100 && fresh.CurrentPlan is not null; frame++)
        {
            var pose = fresh.Tick(.04, KittyAction.Zoomies, true); PoseSafe(pose);
            sawLiftAndStride |= pose.Lift > 8 && Math.Abs(pose.Stride) > .2;
            sawRoll |= Math.Abs(pose.RollDegrees) > 180;
        }
        Check(fresh.CurrentPlan is null, "A generated skill exceeded its finite duration.");
        if (skill == KittyMotionSkill.RunningHop) Check(sawLiftAndStride, "Running-hop did not compose gait and airborne motion together.");
        if (skill is KittyMotionSkill.RollOver or KittyMotionSkill.Tumble) Check(sawRoll, "A rolling skill never rotated through its arc.");
    }
    Check(signatures.Count > 1, "A skill replayed an identical immutable clip instead of varying its plan.");
}

var rewarded = new KittyMotionController(42);
Check(!rewarded.Reinforce(1), "Idle clicks could farm motion learning.");
Check(rewarded.TryPlan(KittyMotionSkill.Hop), "Reward fixture could not start.");
rewarded.Tick(.1, KittyAction.Watch, true); rewarded.Tick(.1, KittyAction.Watch, true);
var before = rewarded.ExportLearning();
Check(rewarded.Reinforce(.7), "Positive contact did not reinforce the current procedural candidate.");
Check(!rewarded.Reinforce(.7), "Repeated contact bypassed the learning rate limit.");
var after = rewarded.ExportLearning();
Check(after.Mastery[1] > before.Mastery[1] && after.Preferences[1] > before.Preferences[1]
    && after.Prototypes.Zip(before.Prototypes, (a, b) => Math.Abs(a - b)).Sum() > 1e-6,
    "Positive contact failed to update actual prototype parameters and preferences.");
var frozen = rewarded.Pose; var completed = rewarded.CompletedPlanCount;
for (var i = 0; i < 20; i++) Check(rewarded.Tick(.1, KittyAction.Zoomies, true, paused: true) == frozen,
    "A paused motion advanced or changed its pose.");
Check(!rewarded.Reinforce(1) && rewarded.CompletedPlanCount == completed, "A hidden/paused skill earned rewards or completion.");
Check(rewarded.Tick(0, KittyAction.Zoomies, true, paused: true) == frozen && !rewarded.Reinforce(1),
    "A zero-time pause allowed contact rewards.");
rewarded.Tick(.1, KittyAction.Watch, false);
Check(rewarded.Pose == KittyMotionPose.Idle && rewarded.CurrentSkill is null
    && rewarded.CompletedPlanCount == completed, "Carry/climb/perch cancellation produced motion or phantom practice.");

var restored = new KittyMotionController(9);
Check(restored.RestoreLearning(learned.Preferences, learned.Mastery, learned.Prototypes, learned.Practice, learned.KnownMask),
    "Valid motion learning could not be restored.");
Check(restored.KnownSkillCount == 6 && restored.ExportLearning().Prototypes.SequenceEqual(learned.Prototypes),
    "Motion persistence lost learned prototypes or discoveries.");
var safeState = restored.ExportLearning(); var bad = (double[])learned.Prototypes.Clone(); bad[3] = double.NaN;
Check(!restored.RestoreLearning(learned.Preferences, learned.Mastery, bad, learned.Practice, learned.KnownMask),
    "A corrupt prototype was accepted.");
Check(restored.ExportLearning().Prototypes.SequenceEqual(safeState.Prototypes), "A corrupt restore partially changed valid learning.");
restored.TryPlan(KittyMotionSkill.Hop); var guarded = restored.Pose;
foreach (var dt in new[] { double.NaN, double.PositiveInfinity, -.1, 0 })
    Check(restored.Tick(dt, KittyAction.Zoomies, true) == guarded, "Invalid delta time advanced a plan.");
restored.Tick(1000, KittyAction.Watch, true);
Check(restored.CurrentPlan is not null, "A delayed frame skipped an entire motion and farmed completion.");
restored.ResetLearning();
Check(restored.KnownSkillCount == 2 && restored.CompletedPlanCount == 0
    && restored.ExportLearning().Practice.All(v => v == 0) && restored.Pose == KittyMotionPose.Idle,
    "Forget/reset retained motion learning or an active pose.");

var a = new KittyMotionController(55); var b = new KittyMotionController(55);
for (var i = 0; i < 2000; i++)
{
    var pa = a.Tick(.05, KittyAction.Zoomies, true, .9, .8, .9);
    var pb = b.Tick(.05, KittyAction.Zoomies, true, .9, .8, .9);
    Check(pa == pb, "Seeded procedural generation is not deterministic.");
    PoseSafe(pa);
}
Check(a.GeneratedPlanCount > 4 && a.KnownSkillCount > 2, "Neural play intent did not produce autonomous practice and growth.");
Console.WriteLine($"PASS {checks} procedural motion checks; six bounded skills, learned prototypes, composition, persistence and feedback guards.");
