// Copyright (c) 2026 Face Tracking Bridge contributors.
// SPDX-License-Identifier: Apache-2.0
namespace BabbleFrameBridge.App;

public enum KittyMotionSkill { RollOver, Hop, BoundingRun, RunningHop, Tumble, StretchWave }

public readonly record struct KittyMotionPose(double Lift, double RollDegrees, double StretchX,
    double StretchY, double Stride, double Tail, double SpeedMultiplier, double Phase,
    bool Active, KittyMotionSkill? Skill)
{
    public static KittyMotionPose Idle => new(0, 0, 1, 1, 0, 0, 1, 0, false, null);
}

public readonly record struct KittyMotionPlan(KittyMotionSkill Skill, double Duration, double Height,
    double Tempo, double RollLead, double Stride, double Tail, int Direction, long Generation);

public sealed record KittyMotionLearning(double[] Preferences, double[] Mastery,
    double[] Prototypes, long[] Practice, int KnownMask);

/// <summary>
/// A bounded procedural motion grammar. The neural mind supplies an intended
/// activity; this controller composes trajectories and learns small prototype
/// preferences from practice and positive contact. It never generates code,
/// clicks controls, changes tracking, or moves a carried/perched/climbing cat.
/// All values are local, finite, and independent of WPF or pointer coordinates.
/// </summary>
public sealed class KittyMotionController
{
    public const int SkillCount = 6, PrototypeWidth = 5, InitialKnownMask = 34;
    public const double MaximumLift = 42, MaximumRoll = 360;
    private readonly Random _random;
    private readonly double[] _preferences = new double[SkillCount];
    private readonly double[] _mastery = new double[SkillCount];
    private readonly double[] _prototypes = new double[SkillCount * PrototypeWidth];
    private readonly long[] _practice = new long[SkillCount];
    private readonly double[] _candidate = new double[PrototypeWidth];
    private int _knownMask = InitialKnownMask;
    private double _age, _nextAttempt = 3, _lastRewardAt = -1, _rewardBudget;
    private bool _paused;
    public KittyMotionPlan? CurrentPlan { get; private set; }
    public KittyMotionSkill? CurrentSkill => CurrentPlan?.Skill;
    public KittyMotionPose Pose { get; private set; } = KittyMotionPose.Idle;
    public long GeneratedPlanCount { get; private set; }
    public long CompletedPlanCount { get; private set; }
    public int KnownSkillCount => Enumerable.Range(0, SkillCount).Count(i => (_knownMask & (1 << i)) != 0);
    public bool IsKnown(KittyMotionSkill skill) => Valid(skill) && (_knownMask & (1 << (int)skill)) != 0;
    public double Mastery(KittyMotionSkill skill) => Valid(skill) ? _mastery[(int)skill] : 0;

    public KittyMotionController(int? seed = null)
    {
        _random = new Random(seed ?? Random.Shared.Next());
        ResetLearning();
    }

    public void ResetLearning()
    {
        Cancel();
        Array.Fill(_preferences, .5); Array.Fill(_mastery, .08);
        Array.Fill(_prototypes, .5); Array.Clear(_practice);
        _mastery[(int)KittyMotionSkill.Hop] = .18;
        _mastery[(int)KittyMotionSkill.StretchWave] = .22;
        _knownMask = InitialKnownMask; GeneratedPlanCount = CompletedPlanCount = 0;
        _nextAttempt = 3;
    }

    public KittyMotionLearning ExportLearning() => new((double[])_preferences.Clone(),
        (double[])_mastery.Clone(), (double[])_prototypes.Clone(), (long[])_practice.Clone(), _knownMask);

    public bool RestoreLearning(double[]? preferences, double[]? mastery, double[]? prototypes,
        long[]? practice, int knownMask = InitialKnownMask)
    {
        if (preferences?.Length != SkillCount || mastery?.Length != SkillCount
            || prototypes?.Length != SkillCount * PrototypeWidth || practice?.Length != SkillCount
            || knownMask is < 0 or > 63 || preferences.Any(v => !double.IsFinite(v))
            || mastery.Any(v => !double.IsFinite(v)) || prototypes.Any(v => !double.IsFinite(v))) return false;
        // Validation precedes every write, so a malformed state cannot partially
        // overwrite the current personal repertoire or interrupt its motion.
        for (var i = 0; i < SkillCount; i++)
        {
            _preferences[i] = Unit(preferences[i]); _mastery[i] = Unit(mastery[i]);
            _practice[i] = Math.Clamp(practice[i], 0, 1_000_000);
        }
        for (var i = 0; i < _prototypes.Length; i++) _prototypes[i] = Unit(prototypes[i]);
        _knownMask = knownMask | InitialKnownMask;
        CompletedPlanCount = _practice.Sum(); Cancel(); return true;
    }

    /// <summary>Pause freezes the current pose; loss of free ground cancels it.</summary>
    public KittyMotionPose Tick(double deltaSeconds, KittyAction action, bool canPerform,
        double energy = .7, double bond = .5, double playfulness = .5, bool paused = false,
        double neuralDrive = .5)
    {
        if (!double.IsFinite(deltaSeconds) || deltaSeconds < 0) return Pose;
        _paused = paused;
        if (paused) return Pose;
        if (!canPerform) { Cancel(); return Pose; }
        if (deltaSeconds == 0) return Pose;
        var dt = Math.Min(deltaSeconds, .1);
        energy = Unit(energy); bond = Unit(bond); playfulness = Unit(playfulness);
        if (action is KittyAction.Sleep or KittyAction.Doze or KittyAction.Startle or KittyAction.Backoff)
        { Cancel(); return Pose; }
        if (CurrentPlan is null)
        {
            _nextAttempt = Math.Max(0, _nextAttempt - dt);
            if (_nextAttempt > 0) return Pose;
            var chosen = Choose(action, energy, bond, playfulness, Unit(neuralDrive));
            if (chosen is null) { _nextAttempt = 1; return Pose; }
            TryPlan(chosen.Value, energy, bond, playfulness);
        }
        if (CurrentPlan is not { } plan) return Pose;
        _age = Math.Min(plan.Duration, _age + dt);
        var phase = Math.Clamp(_age / plan.Duration, 0, 1);
        Pose = Compose(plan, phase);
        if (phase >= 1)
        {
            Complete(plan.Skill);
            CurrentPlan = null; Pose = KittyMotionPose.Idle;
            _nextAttempt = 2.8 + _random.NextDouble() * 4;
        }
        return Pose;
    }

    public bool TryPlan(KittyMotionSkill skill, double energy = .7, double bond = .5, double playfulness = .5)
    {
        if (_paused || CurrentPlan is not null || !IsKnown(skill)) return false;
        energy = Unit(energy); bond = Unit(bond); playfulness = Unit(playfulness);
        if (energy < .2 && skill != KittyMotionSkill.StretchWave) return false;
        var index = (int)skill; var exploration = .04 + .16 * (1 - _mastery[index]);
        for (var j = 0; j < PrototypeWidth; j++)
            _candidate[j] = Unit(_prototypes[index * PrototypeWidth + j] + (_random.NextDouble() * 2 - 1) * exploration);
        var tempo = .8 + .55 * _candidate[0];
        var duration = (skill switch
        {
            KittyMotionSkill.RollOver => 1.65, KittyMotionSkill.Hop => .95,
            KittyMotionSkill.BoundingRun => 1.8, KittyMotionSkill.RunningHop => 1.4,
            KittyMotionSkill.Tumble => 1.9, _ => 2.1
        }) / tempo * (.95 + _random.NextDouble() * .1);
        var height = skill switch
        {
            KittyMotionSkill.Hop => 15 + 17 * _candidate[1],
            KittyMotionSkill.RunningHop => 20 + 22 * _candidate[1],
            KittyMotionSkill.BoundingRun => 5 + 10 * _candidate[1],
            KittyMotionSkill.Tumble => 10 + 18 * _candidate[1], _ => 0
        };
        // Temperament and current neural intent vary plans without changing
        // the safety envelope or the set of available mathematical primitives.
        height *= .75 + .25 * Math.Max(playfulness, energy);
        GeneratedPlanCount = Math.Min(long.MaxValue - 1, GeneratedPlanCount + 1);
        CurrentPlan = new(skill, Math.Clamp(duration, .55, 3), Math.Clamp(height, 0, MaximumLift),
            tempo, .12 + .18 * _candidate[2], .5 + .5 * _candidate[3],
            (.35 + .65 * _candidate[4]) * (.8 + .2 * bond), _random.Next(2) == 0 ? -1 : 1, GeneratedPlanCount);
        _age = 0; _lastRewardAt = -1; _rewardBudget = 1.5;
        Pose = Compose(CurrentPlan.Value, 0); return true;
    }

    /// <summary>Bounded reward-weighted prototype learning from positive contact.</summary>
    public bool Reinforce(double reward)
    {
        if (_paused || CurrentPlan is not { } plan || !double.IsFinite(reward) || reward <= 0
            || _age < .12 || _age - _lastRewardAt < .3 || _rewardBudget <= 0) return false;
        var amount = Math.Min(Math.Clamp(reward, 0, 1), _rewardBudget);
        var index = (int)plan.Skill; _rewardBudget -= amount; _lastRewardAt = _age;
        _preferences[index] = Unit(_preferences[index] + .11 * amount * (1 - _preferences[index]));
        _mastery[index] = Unit(_mastery[index] + .07 * amount * (1 - _mastery[index]));
        Adapt(index, .12 * amount);
        return true;
    }

    public void Cancel()
    {
        CurrentPlan = null; Pose = KittyMotionPose.Idle; _age = 0;
        _lastRewardAt = -1; _rewardBudget = 0; _paused = false;
        _nextAttempt = Math.Max(_nextAttempt, 1);
    }

    private KittyMotionSkill? Choose(KittyAction action, double energy, double bond, double playfulness, double drive)
    {
        Span<double> weights = stackalloc double[SkillCount]; var total = 0d;
        for (var i = 0; i < SkillCount; i++)
        {
            var skill = (KittyMotionSkill)i;
            if (!IsKnown(skill) || energy < .2 && skill != KittyMotionSkill.StretchWave) continue;
            var compatibility = skill switch
            {
                KittyMotionSkill.RollOver => action is KittyAction.Roll or KittyAction.Flop ? 2.6 : .04,
                KittyMotionSkill.Hop => action is KittyAction.Pounce or KittyAction.Chase or KittyAction.Zoomies or KittyAction.TailChase ? 1.5 : .04,
                KittyMotionSkill.BoundingRun => action is KittyAction.Walk or KittyAction.Chase or KittyAction.Zoomies or KittyAction.TailChase ? 2.2 : .03,
                KittyMotionSkill.RunningHop => action is KittyAction.Chase or KittyAction.Zoomies or KittyAction.Pounce ? 2.4 : .02,
                KittyMotionSkill.Tumble => action is KittyAction.Roll or KittyAction.Zoomies or KittyAction.Flop ? 1.4 : .015,
                _ => action is KittyAction.Stretch or KittyAction.Yawn or KittyAction.Groom or KittyAction.Watch or KittyAction.LookAround ? 1.4 : .06
            };
            if (action is KittyAction.Grumpy or KittyAction.Peek) compatibility *= .1;
            var exploratory = skill == KittyMotionSkill.StretchWave ? .4 + .4 * bond : .3 + .7 * playfulness;
            var complexity = skill is KittyMotionSkill.RunningHop or KittyMotionSkill.Tumble
                ? .2 + .8 * drive : skill == KittyMotionSkill.BoundingRun ? .5 + .5 * drive : 1 - .2 * drive;
            weights[i] = compatibility * (.45 + _preferences[i]) * (.6 + .4 * _mastery[i])
                * exploratory * complexity;
            total += weights[i];
        }
        if (total <= 0) return null;
        var pick = _random.NextDouble() * total;
        for (var i = 0; i < SkillCount; i++) if ((pick -= weights[i]) <= 0 && weights[i] > 0) return (KittyMotionSkill)i;
        return null;
    }

    private void Complete(KittyMotionSkill skill)
    {
        var index = (int)skill;
        _practice[index] = Math.Min(1_000_000, _practice[index] + 1);
        CompletedPlanCount = Math.Min(long.MaxValue - 1, CompletedPlanCount + 1);
        _mastery[index] = Unit(_mastery[index] + .045 * (1 - _mastery[index]));
        Adapt(index, .02);
        var practiced = _practice.Sum();
        // A finite grammar can expand through practice without requiring the
        // owner to keep clicking or waiting behind an unbounded unlock ladder.
        if (practiced >= 3) Discover(KittyMotionSkill.RollOver, KittyMotionSkill.StretchWave, KittyMotionSkill.Hop);
        if (practiced >= 5) Discover(KittyMotionSkill.BoundingRun, KittyMotionSkill.Hop, KittyMotionSkill.StretchWave);
        if (practiced >= 9) Discover(KittyMotionSkill.RunningHop, KittyMotionSkill.Hop, KittyMotionSkill.BoundingRun);
        if (practiced >= 15) Discover(KittyMotionSkill.Tumble, KittyMotionSkill.RollOver, KittyMotionSkill.RunningHop);
    }

    private void Adapt(int index, double rate)
    {
        for (var j = 0; j < PrototypeWidth; j++)
        {
            var slot = index * PrototypeWidth + j;
            _prototypes[slot] = Unit(_prototypes[slot] + rate * (_candidate[j] - _prototypes[slot]));
        }
    }

    private void Discover(KittyMotionSkill skill, KittyMotionSkill first, KittyMotionSkill second)
    {
        if (IsKnown(skill)) return;
        var index = (int)skill; _knownMask |= 1 << index;
        _mastery[index] = .05 + .1 * (_mastery[(int)first] + _mastery[(int)second]);
        for (var j = 0; j < PrototypeWidth; j++)
            _prototypes[index * PrototypeWidth + j] = Unit((_prototypes[(int)first * PrototypeWidth + j]
                + _prototypes[(int)second * PrototypeWidth + j]) * .5 + (_random.NextDouble() - .5) * .08);
    }

    private static KittyMotionPose Compose(KittyMotionPlan plan, double p)
    {
        var envelope = Math.Sin(Math.PI * p); var smooth = p * p * (3 - 2 * p);
        var stride = Math.Sin(p * Math.PI * (5 + 3 * plan.Stride)) * envelope * plan.Stride;
        var lift = 0d; var roll = 0d; var speed = 1d;
        switch (plan.Skill)
        {
            case KittyMotionSkill.Hop: lift = plan.Height * envelope; break;
            case KittyMotionSkill.BoundingRun:
                lift = plan.Height * Math.Abs(Math.Sin(p * Math.PI * 3)) * envelope;
                speed = 1 + .35 * envelope; break;
            case KittyMotionSkill.RunningHop:
                lift = plan.Height * envelope; speed = 1 + .4 * envelope; break;
            case KittyMotionSkill.RollOver:
                var rolling = Math.Clamp((p - plan.RollLead) / (1 - 2 * plan.RollLead), 0, 1);
                roll = plan.Direction * MaximumRoll * rolling * rolling * (3 - 2 * rolling);
                speed = 1 - .35 * envelope; break;
            case KittyMotionSkill.Tumble:
                lift = plan.Height * envelope; roll = plan.Direction * MaximumRoll * smooth;
                speed = 1 + .2 * envelope; break;
            case KittyMotionSkill.StretchWave: speed = 1 - .15 * envelope; break;
        }
        var stretch = plan.Skill == KittyMotionSkill.StretchWave ? .16 : .07;
        var stretchX = 1 + stretch * envelope * Math.Cos(p * Math.PI * 2);
        var stretchY = 1 - stretch * envelope * .65;
        return new(Math.Clamp(lift, 0, MaximumLift), Math.Clamp(roll, -MaximumRoll, MaximumRoll),
            Math.Clamp(stretchX, .8, 1.2), Math.Clamp(stretchY, .8, 1.2), Math.Clamp(stride, -1, 1),
            Math.Sin(p * Math.PI * (2 + plan.Tempo)) * envelope * plan.Tail,
            Math.Clamp(speed, .6, 1.4), p, true, plan.Skill);
    }

    private static bool Valid(KittyMotionSkill skill) => (int)skill is >= 0 and < SkillCount;
    private static double Unit(double value) => double.IsFinite(value) ? Math.Clamp(value, 0, 1) : 0;
}
