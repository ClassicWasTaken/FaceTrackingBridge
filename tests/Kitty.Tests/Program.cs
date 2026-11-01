// Copyright (c) 2026 Frame Bridge contributors.
// SPDX-License-Identifier: Apache-2.0
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using BabbleFrameBridge.App;

var checks=0;
void Check(bool pass,string message) { if(!pass) throw new Exception(message); checks++; }
var arrays = typeof(KittyModel).GetFields(BindingFlags.Static | BindingFlags.NonPublic)
    .Where(field => field.FieldType == typeof(float[])).ToDictionary(field => field.Name, field => (float[])field.GetValue(null)!);
var trainedParameterCount = arrays.Where(pair => pair.Key != "PositionEncoding").Sum(pair => pair.Value.Length);
Check(trainedParameterCount == KittyBrain.ParameterCount,
    "Transformer trainable parameter inventory changed.");
Check(KittyBrain.ParameterCount == 1070888 && trainedParameterCount is >= 1000000 and <= 1200000
    && arrays.Sum(pair => pair.Value.Length) * sizeof(float) < 5000000,
    "The original float32 Transformer exceeded its lightweight weight budget.");
Check(arrays.Values.All(values => values.All(float.IsFinite)), "Transformer weights contain invalid values.");
Check(KittyBrain.LayerCount == 2 && KittyBrain.HeadCount == 8 && KittyBrain.ModelWidth == 256 && KittyBrain.FeedForwardWidth == 512 && KittyBrain.ContextLength == 32,
    "The causal multi-head Transformer architecture changed.");
Check(arrays["Block0QkvWeights"].Length == 256 * 768 && arrays["Block1QkvWeights"].Length == 256 * 768
    && arrays["PositionEncoding"].Length == 32 * 256, "Attention projections or positional encoding are absent.");
Check(Enum.GetValues<KittyAction>().Length == 35 && (int)KittyAction.Stretch == 9 && (int)KittyAction.Circle == 24 && (int)KittyAction.Startle == 34,
    "Existing actions were renumbered or the cat repertoire is incomplete.");

var calm=new KittyObservation(.85,.08,0,0,0,0,0,0,0,.55,0,.25);
if (args.Contains("--benchmark-only")) { Benchmark(); return; }
if (args.Contains("--hunt-only")) { CheckPredatoryPlay(); Console.WriteLine($"PASS {checks} targeted predatory-play checks."); return; }
double Mass(KittyObservation observation, params KittyAction[] actions)
{
    var probabilities = new KittyBrain().Predict(observation);
    return actions.Sum(action => probabilities[(int)action]);
}
var playFamily = new[] { KittyAction.Zoomies, KittyAction.Chase, KittyAction.Swat, KittyAction.Stalk, KittyAction.Pounce, KittyAction.ButtWiggle, KittyAction.TailChase };
var spaceFamily = new[] { KittyAction.Grumpy, KittyAction.Backoff, KittyAction.Startle, KittyAction.Peek };
var restFamily = new[] { KittyAction.Sleep, KittyAction.Doze, KittyAction.Loaf, KittyAction.Knead, KittyAction.Circle };
Check(Mass(calm with { Stimulation=.95,CursorNear=1,CursorMotion=1 }, playFamily) > Mass(calm, playFamily) + .15,
    "Strong play stimulation did not increase playful policy preference.");
Check(Mass(calm with { Irritation=.95,ClickPressure=1,Stimulation=.7 }, spaceFamily) > Mass(calm, spaceFamily) + .15,
    "Pestering did not increase space-seeking policy preference.");
Check(Mass(calm with { Energy=.25,Sleepiness=.85,Idle=1 }, restFamily) > Mass(calm, restFamily) + .15,
    "Quiet exhaustion did not increase resting policy preference.");

var a=new KittyBrain(); var b=new KittyBrain();
for(var i=0;i<12;i++) {
    a.Predict(calm with { CursorNear=1,CursorMotion=1,Stimulation=.4 });
    b.Predict(calm);
}
var afterPlay=a.Predict(calm); var afterQuiet=b.Predict(calm);
Check(!a.SequenceHistory.SequenceEqual(b.SequenceHistory),"Different recent histories produced identical Transformer context.");
Check(afterPlay.Zip(afterQuiet,(x,y)=>Math.Abs(x-y)).Sum()>.001,"Past experience did not influence the same present observation.");
a.ResetMemory(); b.ResetMemory();
Check(a.Predict(calm).SequenceEqual(b.Predict(calm)),"Resetting temporal memory was not deterministic.");

foreach(var o in new[] { calm,default,new KittyObservation(double.NaN,double.PositiveInfinity,-99,100,0,0,0,0,0,0,0,0) }) {
    var p=a.Predict(o);
    Check(p.All(v=>double.IsFinite(v)&&v>=0&&v<=1)&&Math.Abs(p.Sum()-1)<1e-12,"Behavior output is not a bounded distribution.");
    var gaze=a.AttentionProbabilities.ToArray();
    Check(gaze.All(v=>double.IsFinite(v)&&v>=0&&v<=1)&&Math.Abs(gaze.Sum()-1)<1e-12,"Attention output is not a bounded distribution.");
}

// A real causal mask must protect ALL earlier positions, not just a single
// final prediction. Injecting a future impulse cannot change a prefix token.
var normalized = new float[6 * KittyBrain.InputCount];
for (var index = 0; index < normalized.Length; index++) normalized[index] = (float)Math.Sin(index * .37);
var transformer = new KittyTransformer();
var prefixActions = new double[KittyBrain.ActionCount]; var futureActions = new double[KittyBrain.ActionCount];
var prefixAttention = new double[KittyBrain.AttentionCount]; var futureAttention = new double[KittyBrain.AttentionCount];
transformer.Evaluate(normalized.AsSpan(0, 3 * KittyBrain.InputCount), 3, prefixActions, prefixAttention);
transformer.Evaluate(normalized, 6, futureActions, futureAttention, outputPosition: 2);
Check(prefixActions.SequenceEqual(futureActions) && prefixAttention.SequenceEqual(futureAttention),
    "A future observation leaked through the causal attention mask.");

var windowA = new KittyBrain(); var windowB = new KittyBrain();
windowA.Predict(calm with { CursorMotion = 1, Irritation = 1 }); windowB.Predict(calm);
for (var index = 0; index < KittyBrain.ContextLength; index++) { windowA.Predict(calm); windowB.Predict(calm); }
Check(windowA.HistoryCount == KittyBrain.ContextLength && windowA.SequenceHistory.SequenceEqual(windowB.SequenceHistory),
    "The rolling context retained observations beyond its32-token capacity.");
Check(windowA.Predict(calm).SequenceEqual(windowB.Predict(calm)), "Expired context continued to influence a new prediction.");
windowA.ResetHistory();
Check(windowA.HistoryCount == 0 && windowA.MemoryActivity == 0 && windowA.AttentionProbabilities.ToArray().All(value => value == 0),
    "Reset did not clear sequence and attention memory.");
try { windowA.Predict(calm, new double[KittyBrain.ActionCount - 1]); Check(false, "An undersized output buffer was accepted."); }
catch (ArgumentException) { Check(windowA.HistoryCount == 0, "A rejected output buffer mutated context."); }
var sanitizedA = new KittyBrain(); var sanitizedB = new KittyBrain();
Check(sanitizedA.Predict(calm with { Energy = double.NaN, Sleepiness = double.PositiveInfinity, Irritation = -3, Stimulation = 4 })
    .SequenceEqual(sanitizedB.Predict(calm with { Energy = 0, Sleepiness = 0, Irritation = 0, Stimulation = 1 })),
    "Nonfinite/out-of-range inputs were not normalized consistently.");
var encoded = sanitizedA.ExportContext(); var loaded = new KittyBrain(); loaded.RestoreContext(encoded);
Check(loaded.HistoryCount == sanitizedA.HistoryCount && encoded.Length == KittyBrain.InputCount
    && encoded.All(value => value is >= 0 and <= 65535), "Persisted context is not a bounded quantized observation sequence.");
Check(sanitizedA.Predict(calm).Zip(loaded.Predict(calm), (left, right) => Math.Abs(left - right)).Max() < .001,
    "Restored context changed the next policy beyond quantization tolerance.");
var fullContext = new KittyBrain();
for (var index = 0; index < KittyBrain.ContextLength; index++)
    fullContext.Predict(calm with { CursorNear = index % 3 / 2d, CursorMotion = index % 5 / 4d, Novelty = index % 7 / 6d });
var fullRestored = new KittyBrain(); fullRestored.RestoreContext(fullContext.ExportContext());
Check(fullRestored.HistoryCount == KittyBrain.ContextLength
    && fullContext.Predict(calm).Zip(fullRestored.Predict(calm), (left, right) => Math.Abs(left - right)).Max() < .001,
    "A full diverse saved context did not influence the next decision within quantization tolerance.");
foreach (var invalid in new[] { new[] { 5 }, Enumerable.Repeat(0, 33 * 22).ToArray(), Enumerable.Repeat(-1, 22).ToArray(), Enumerable.Repeat(65536, 22).ToArray() })
{ loaded.RestoreContext(invalid); Check(loaded.HistoryCount == 0, "Invalid persisted Transformer context was accepted."); }

using (var golden = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "transformer-golden.json"))))
{
    var maximumError = 0d; var cases = 0;
    var learned = new Dictionary<string, (double[] Behavior, double[] Attention, double[] LastObservation)>();
    foreach (var example in golden.RootElement.GetProperty("cases").EnumerateArray())
    {
        var brain = new KittyBrain(); double[] actual = [];
        foreach (var observation in example.GetProperty("observations").EnumerateArray())
        {
            var values = observation.EnumerateArray().Select(value => value.GetDouble()).ToArray();
            Check(values.Length == KittyBrain.InputCount, "A golden observation has a wrong feature count.");
            actual = brain.Predict(new(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
                values[8], values[9], values[10], values[11], values[12], values[13], values[14], values[15], values[16], values[17],
                values[18], values[19], values[20], values[21]));
        }
        var expected = example.GetProperty("behavior").EnumerateArray().Select(value => value.GetDouble()).ToArray();
        var expectedAttention = example.GetProperty("attention").EnumerateArray().Select(value => value.GetDouble()).ToArray();
        Check(expected.Length == KittyBrain.ActionCount && expectedAttention.Length == KittyBrain.AttentionCount, "A golden head has the wrong size.");
        maximumError = Math.Max(maximumError, actual.Zip(expected, (left, right) => Math.Abs(left - right)).Max());
        maximumError = Math.Max(maximumError, brain.AttentionProbabilities.ToArray().Zip(expectedAttention, (left, right) => Math.Abs(left - right)).Max());
        learned[example.GetProperty("name").GetString()!] = (actual, brain.AttentionProbabilities.ToArray(),
            example.GetProperty("observations").EnumerateArray().Last().EnumerateArray().Select(value => value.GetDouble()).ToArray());
        cases++;
    }
    foreach (var name in new[] { "play", "gentle", "pester", "novel", "repeated", "gradual" })
        Check(learned[name].LastObservation.SequenceEqual(learned["quiet"].LastObservation), "A temporal policy probe changed the present observation.");
    double LearnedMass(string name, params KittyAction[] actions) => actions.Sum(action => learned[name].Behavior[(int)action]);
    Check(LearnedMass("play", KittyAction.Groom, KittyAction.PawLick, KittyAction.FaceWash)
        > LearnedMass("quiet", KittyAction.Groom, KittyAction.PawLick, KittyAction.FaceWash) + .01,
        "Remembered play did not increase learned post-play grooming preference at the same present state.");
    Check(LearnedMass("gentle", KittyAction.Rub, KittyAction.SlowBlink, KittyAction.Knead, KittyAction.Loaf)
        > LearnedMass("pester", KittyAction.Rub, KittyAction.SlowBlink, KittyAction.Knead, KittyAction.Loaf) + .05,
        "Gentle and pestering histories did not produce different learned affiliative preferences.");
    Check(LearnedMass("pester", KittyAction.Grumpy, KittyAction.Backoff, KittyAction.Startle, KittyAction.Peek)
        > LearnedMass("quiet", KittyAction.Grumpy, KittyAction.Backoff, KittyAction.Startle, KittyAction.Peek) + .005,
        "Earlier pestering did not retain a learned space-seeking preference after the input returned to calm.");
    Check(learned["novel"].Attention[(int)KittyAttention.Memory] > learned["quiet"].Attention[(int)KittyAttention.Memory] + .08,
        "A remembered novelty did not affect learned attention at the identical present state.");
    Check(learned["repeated"].Attention[(int)KittyAttention.Environment] > learned["novel"].Attention[(int)KittyAttention.Environment] + .08,
        "Repeated unchanging stimulation did not differ from a novel encounter in learned attention.");
    Check(cases >= 6 && maximumError < .0001, $"Python/C# Transformer parity failed: {maximumError:G4}.");
    Console.WriteLine($"PASS {cases} Python/C# golden sequences; maximum probability error {maximumError:G4}.");
}

var affectionate = new KittyMind(51); var distant = affectionate.CaptureMemory();
for (var index = 0; index < 20; index++) { affectionate.Tick(.25, default); affectionate.Pet(.5); }
var friendly = affectionate.CaptureMemory();
Check(friendly.Trust > distant.Trust && friendly.Affection > distant.Affection && friendly.ContactMemory == 1,
    "Gentle petting did not build aggregate familiarity.");
var pokeMind = new KittyMind(51);
for (var index = 0; index < 14; index++) { pokeMind.Poke(); pokeMind.Tick(.1, new(1, 0, 0, 0, 1)); }
Check(pokeMind.Irritation > .7 && pokeMind.Trust < distant.Trust, "Repeated poking did not request personal space.");
var restored = new KittyMind(99); restored.RestoreMemory(friendly);
Check(restored.Trust == friendly.Trust && restored.Affection == friendly.Affection && restored.MemoryActivity > 0
    && restored.CaptureMemory().Context!.SequenceEqual(friendly.Context!),
    "Restoring aggregate familiarity or bounded Transformer context changed it.");
var copy = affectionate.CaptureMemory(); var preferenceBefore = affectionate.CaptureMemory().Preferences![0];
copy.Preferences![0] = -1;
Check(affectionate.CaptureMemory().Preferences![0] == preferenceBefore, "Captured memory aliased mutable in-session preferences.");
restored.RestoreMemory(new(Affection: double.NaN, Trust: double.PositiveInfinity, Habituation: -4, SocialNeed: 10,
    PlayMemory: -1, RestMemory: 8, ContactMemory: double.NaN, Preferences: Enumerable.Repeat(double.NaN, 35).ToArray()));
var bounded = restored.CaptureMemory();
Check(bounded.Affection == .55 && bounded.Trust == .65 && bounded.Habituation == 0 && bounded.SocialNeed == 1
    && bounded.PlayMemory == 0 && bounded.RestMemory == 1 && bounded.ContactMemory == 0 && bounded.Preferences!.All(value => value == 1),
    "Corrupt aggregate memory introduced invalid motivation or model inputs.");
Check(friendly.CoreMemories!.Any(memory => memory.Kind == KittyCoreMemoryKind.FirstGentlePet) && friendly.Bond > 0,
    "Sustained gentle contact did not form a first positive core memory or bond.");
var spam = new KittyMind(51);
for (var index = 0; index < 10000; index++) { spam.Pet(); spam.Poke(); }
Check(spam.PositiveContactSeconds <= .25 && spam.CoreMemoryCount == 0 && spam.CaptureMemory().Bond < .001,
    "Callbacks without elapsed active time farmed a bond or positive core memories.");
var carried = new KittyMind(28); carried.OnPickup();
for (var index = 0; index < 8; index++) carried.Tick(.1, default);
carried.EndPickup(.9);
Check(carried.CaptureMemory().CoreMemories!.Any(memory => memory.Kind == KittyCoreMemoryKind.CalmPickup),
    "A calm sustained pickup did not become a positive episode.");
var canceledCarry = new KittyMind(28); canceledCarry.OnPickup(); canceledCarry.EndPickup(.9);
Check(canceledCarry.CoreMemoryCount == 0, "A zero-duration pickup manufactured a core memory.");
var personalityA = new KittyMind(43); var personalityB = new KittyMind(43); var personalityC = new KittyMind(44);
Check(personalityA.TemperamentSummary == personalityB.TemperamentSummary && personalityA.Playfulness == personalityB.Playfulness
    && personalityA.Playfulness != personalityC.Playfulness, "Persistent individual temperament was not reproducible or individual.");
var freshA = new KittyMind(15); var freshB = new KittyMind(16); var originalTemperament = freshA.Playfulness;
freshA.RestoreMemory(new()); freshB.RestoreMemory(new());
Check(freshA.Playfulness == originalTemperament && freshA.Playfulness != freshB.Playfulness,
    "A new or legacy memory without an identity erased the generated individual temperament.");
var personSaved = affectionate.CaptureMemory(); var otherSession = new KittyMind(742); otherSession.RestoreMemory(personSaved);
Check(otherSession.CaptureMemory().IdentitySeed == personSaved.IdentitySeed && otherSession.Playfulness == personSaved.Playfulness
    && otherSession.CoreMemoryCount == personSaved.CoreMemories!.Length && otherSession.Bond == personSaved.Bond,
    "A returning companion lost its identity, temperament, bond, or positive episodes.");
var cooloff = new KittyMind(39);
for (var index = 0; index < 14; index++) { cooloff.Poke(); cooloff.Tick(.1, new(1, 0, 0, 0, 1)); }
for (var index = 0; index < 200; index++) { cooloff.Tick(.125, default); if (index % 2 == 0) cooloff.Pet(.6); }
Check(cooloff.Irritation < .1 && cooloff.CaptureMemory().CoreMemories!.Any(memory => memory.Kind == KittyCoreMemoryKind.ComfortAfterStartle),
    "Space and gentle reassurance did not recover irritation or form a comforting episode.");
var contactCadence = new KittyMind(91);
for (var index = 0; index < 1000; index++) { contactCadence.Pet(.01); contactCadence.Tick(.01, default); }
Check(contactCadence.InferenceCount is >= 49 and <= 52, "Repeated contact exceeded the5Hz Transformer context cadence.");

// Calm positive feedback trains a small personal head adapter by an actual
// cross-entropy SGD step. Base model arrays stay unchanged and pokes are not labels.
var learner = new KittyMind(502); learner.Tick(.25, new(UserPresence: 1));
for (var index = 0; index < 3; index++) { learner.OnGentleContact(.25, .5); learner.Tick(.25, new(UserPresence: 1)); }
var behaviorBeforeLearning = learner.BehaviorProbabilities.ToArray(); var reinforcedAction = learner.Action;
learner.OnGentleContact(.25, .5);
Check(learner.LearningUpdates == 1 && learner.PolicyBiases[(int)reinforcedAction] > 0,
    "Calm positive feedback did not train the personal policy head.");
var expectedGradient = .01 * (1 - behaviorBeforeLearning[(int)reinforcedAction]);
Check(Math.Abs(learner.PolicyBiases[(int)reinforcedAction] - expectedGradient) < 1e-10,
    "The online adapter did not apply the declared cross-entropy gradient.");
Check(learner.BehaviorProbabilities[(int)reinforcedAction] > behaviorBeforeLearning[(int)reinforcedAction]
    && Math.Abs(learner.BehaviorProbabilities.ToArray().Sum() - 1) < 1e-12,
    "Learning did not influence the actual normalized model probabilities.");
var learnedBias = learner.PolicyBiases.ToArray();
for (var index = 0; index < 10000; index++) learner.Pet(1);
Check(learner.LearningUpdates == 1 && learner.PolicyBiases.SequenceEqual(learnedBias), "Callback spam manufactured personal gradient steps.");
for (var index = 0; index < 14; index++) { learner.Poke(); learner.Tick(.1, new(1, 0, 0, 0, 1)); }
Check(learner.LearningUpdates == 1 && learner.PolicyBiases.SequenceEqual(learnedBias), "Pokes were treated as explicit training dislikes.");
var savedLearning = learner.CaptureMemory(); var returnedLearner = new KittyMind(503); returnedLearner.RestoreMemory(savedLearning);
Check(returnedLearner.LearningUpdates == 1 && returnedLearner.PolicyBiases.SequenceEqual(learnedBias), "Personal learned weights did not survive a memory restore.");
savedLearning.PolicyBiases![0] = 99;
Check(learner.PolicyBiases[0] != 99, "Saved personal learning arrays alias the active model adapter.");
returnedLearner.RestoreMemory(new(PolicyBiases: Enumerable.Repeat(90d, 40).ToArray(), LearningUpdates: long.MaxValue));
Check(returnedLearner.PolicyBiases.ToArray().All(value => value == .8) && returnedLearner.LearningUpdates == 1_000_000_000,
    "Loaded personal learning escaped its bounded capacity.");
returnedLearner.RestoreMemory(new(PolicyBiases: Enumerable.Repeat(double.NaN, 40).ToArray()));
Check(returnedLearner.PolicyBiases.ToArray().All(value => value == 0) && returnedLearner.LearningUpdates == 0,
    "Invalid personal weights introduced nonfinite inference or a bogus update count.");
returnedLearner.RestoreMemory(new());
Check(returnedLearner.PolicyBiases.ToArray().All(value => value == 0) && returnedLearner.LearningUpdates == 0,
    "An older compatible memory did not default to a neutral personal adapter.");

static async Task WaitForInference(KittyMind mind)
{
    var started = Stopwatch.StartNew();
    while (mind.IsInferenceRunning)
    { if (started.Elapsed > TimeSpan.FromSeconds(10)) throw new TimeoutException("The bounded policy worker did not finish."); await Task.Delay(2); }
}
using (var releaseInference = new ManualResetEventSlim(false))
using (var enteredInference = new ManualResetEventSlim(false))
{
    var asynchronous = new KittyMind(604, backgroundInference: true);
    asynchronous.BeforeBackgroundInference = () => { enteredInference.Set(); if (!releaseInference.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException("Test worker gate expired."); };
    asynchronous.Tick(.125, default);
    Check(enteredInference.Wait(TimeSpan.FromSeconds(5)), "Background inference did not start independently.");
    var uiClock = Stopwatch.StartNew();
    for (var index = 0; index < 64; index++) asynchronous.Tick(.125, new(index % 2, 0, 0, 0, 1));
    uiClock.Stop();
    Check(asynchronous.InferenceCount == 1 && asynchronous.IsInferenceRunning && uiClock.Elapsed < TimeSpan.FromMilliseconds(100),
        "A blocked policy worker delayed the controller or created a job backlog.");
    Check(asynchronous.CaptureMemory().Context is { Length: 704 }, "Live context could not be saved safely while a policy worker was pending.");
    releaseInference.Set(); await WaitForInference(asynchronous);
    asynchronous.BeforeBackgroundInference = null; asynchronous.Tick(0, default);
    Check(asynchronous.AcceptedInferenceCount == 0 && asynchronous.DroppedInferenceCount == 1,
        "A stale policy result was accepted after a long worker delay.");
    asynchronous.Tick(.2, default); await WaitForInference(asynchronous); asynchronous.Tick(0, default);
    Check(asynchronous.AcceptedInferenceCount == 1 && asynchronous.PolicyError is null,
        "A stale worker prevented the next current-context prediction from recovering.");
}
using (var releaseOldGeneration = new ManualResetEventSlim(false))
{
    var reloaded = new KittyMind(605, backgroundInference: true);
    reloaded.BeforeBackgroundInference = () => releaseOldGeneration.Wait(TimeSpan.FromSeconds(10));
    reloaded.Tick(.125, default); reloaded.RestoreMemory(new());
    releaseOldGeneration.Set(); await WaitForInference(reloaded);
    reloaded.BeforeBackgroundInference = null; reloaded.Tick(0, default);
    Check(reloaded.DroppedInferenceCount == 1 && reloaded.AcceptedInferenceCount == 0,
        "A policy result from before reload crossed the memory generation boundary.");
    await WaitForInference(reloaded); reloaded.Tick(0, default);
    Check(reloaded.AcceptedInferenceCount == 1, "A reloaded context could not produce a new valid background result.");
}
var failingPolicy = new KittyMind(606, backgroundInference: true) { BeforeBackgroundInference = () => throw new InvalidOperationException("Synthetic policy failure") };
failingPolicy.Tick(.125, default); await WaitForInference(failingPolicy); failingPolicy.Tick(0, default);
Check(failingPolicy.PolicyError?.Contains("Synthetic policy failure") == true && failingPolicy.AcceptedInferenceCount == 0,
    "A background inference error was lost or admitted invalid probabilities.");
failingPolicy.BeforeBackgroundInference = null; failingPolicy.Tick(.2, default); await WaitForInference(failingPolicy); failingPolicy.Tick(0, default);
Check(failingPolicy.PolicyError is null && failingPolicy.AcceptedInferenceCount == 1, "A background error could not recover without blocking the app.");

Console.WriteLine($"PASS {checks} Transformer policy, memory, online learning and asynchronous safety checks.");
if (args.Contains("--policy-only")) return;

var attentionMind=new KittyMind(11);
var cursorTicks=0; var otherTicks=0; var focusModes=new HashSet<KittyAttention>();
for(var i=0;i<1200;i++) {
    attentionMind.Tick(.125,new(.8,0,0,0,1)); focusModes.Add(attentionMind.Attention);
    if(attentionMind.Attention==KittyAttention.Cursor) cursorTicks++; else otherTicks++;
}
Check(otherTicks>cursorTicks,"A stationary cursor monopolizes attention.");
Check(attentionMind.Habituation>.7,"Repeated unchanging cursor presence did not habituate.");
Check(focusModes.Contains(KittyAttention.Environment)&&focusModes.Count>=3,"The kitty did not have multiple autonomous attention targets.");

var wave=new KittyMind(42); var waveActions=new HashSet<KittyAction>();
for(var i=0;i<400;i++) { wave.Tick(.125,new(1,1,0,0,1)); waveActions.Add(wave.Action); }
Check(waveActions.Contains(KittyAction.Zoomies),"Near-face waving did not produce zoomies.");
Check(wave.Energy<.8,"Vigorous activity did not create fatigue.");
for(var i=0;i<800;i++) wave.Tick(.125,default);
Check(wave.Stimulation<.02 && wave.Action!=KittyAction.Zoomies,"The kitty could not settle after stimulation stopped.");
Check(wave.InferenceCount>wave.DecisionCount&&wave.InferenceCount<780,"The Transformer policy is not running at a bounded independent thinking cadence.");

var annoyed=new KittyMind(7); var grumpy=false;
for(var i=0;i<14;i++) { annoyed.Pester(); annoyed.Tick(.1,new(1,0,0,0,1)); grumpy|=annoyed.Action==KittyAction.Grumpy; }
Check(annoyed.Irritation>.7&&annoyed.Trust<.6,"Repeated pestering did not affect irritation and trust.");
for(var i=0;i<80;i++) { annoyed.Tick(.125,new(1,0,0)); grumpy|=annoyed.Action==KittyAction.Grumpy; }
Check(grumpy,"The pestered kitty never displayed annoyance.");
for(var i=0;i<480;i++) annoyed.Tick(.125,default);
Check(annoyed.Irritation<.01,"Irritation did not recover after giving the kitty space.");

var home=new KittyMind(21); var homeActions=new HashSet<KittyAction>(); var transitions=new HashSet<(KittyAction,KittyAction)>();
var last=home.Action;
for(var i=0;i<8000;i++) {
    home.Tick(.125,default); homeActions.Add(home.Action);
    if(home.Action!=last) { transitions.Add((last,home.Action)); last=home.Action; }
}
Check(homeActions.Contains(KittyAction.Groom)&&homeActions.Contains(KittyAction.Sleep),"A quiet house kitty failed to groom and nap.");
Check(homeActions.Contains(KittyAction.Knead)&&homeActions.Contains(KittyAction.Loaf),"The quiet kitty never prepared and used a resting pose.");
Check(homeActions.Contains(KittyAction.Yawn)&&homeActions.Contains(KittyAction.Stretch),"The kitty never yawned or stretched after resting.");
Check(homeActions.Contains(KittyAction.SlowBlink)&&homeActions.Contains(KittyAction.LookAround),"Relaxed blinking and independent observation are absent.");
Check(!homeActions.Any(x=>x is KittyAction.Chase or KittyAction.Swat or KittyAction.Stalk or KittyAction.Pounce or KittyAction.SmackPanel or KittyAction.Scratch),
    "The kitty targeted a nonexistent cursor or surface.");
Check(transitions.Contains((KittyAction.Sleep,KittyAction.Stretch)),"The nap-to-wake stretch sequence was not observed.");
Check(transitions.Contains((KittyAction.Circle,KittyAction.Knead))||transitions.Contains((KittyAction.Knead,KittyAction.Loaf)),
    "The settle-to-rest sequence was not observed.");

Console.WriteLine("PASS autonomous attention, fatigue, household grooming and full rest/wake routines.");
CheckPredatoryPlay();
var vocalActions=new HashSet<KittyAction>();
for(var seed=0;seed<12;seed++) {
    var social=new KittyMind(seed);
    for(var i=0;i<2400;i++) social.Tick(.125,default);
    social.Pester();
    for(var i=0;i<240;i++) { social.Tick(.125,new(.6,.2,0,0,1)); vocalActions.Add(social.Action); }
}
Check(vocalActions.Contains(KittyAction.Chirp)||vocalActions.Contains(KittyAction.Meow),"Social/play contexts never selected a vocal behavior.");

Console.WriteLine("PASS predatory play continuity and varied social vocalization contexts.");
Check(home.CompletedRoutineCount>0,"Autonomous home routines never finish.");
Check(homeActions.Contains(KittyAction.PawLick)&&homeActions.Contains(KittyAction.FaceWash),
    "Grooming never unfolds into distinct paw and face washing stages.");
Check(homeActions.Contains(KittyAction.Doze)&&homeActions.Contains(KittyAction.EarFlick),
    "Quiet home behavior lacks light dozing and small attention reactions.");

KittyMind FindHunt(int seed)
{
    var mind=new KittyMind(seed);
    for(var i=0;i<6000;i++)
    {
        var phase=i%200;
        mind.Tick(.125,phase<70 ? new(.85,.55,1,phase<4?1:0,1) : new(.4,0,1,0,1));
        if(mind.Routine==KittyRoutine.Hunt&&mind.Action==KittyAction.Stalk&&mind.ActionAge<.2) return mind;
    }
    throw new Exception("The neural behavior selector never started a hunt episode.");
}
var interruptedHunt=FindHunt(23);
Check(interruptedHunt.Action==KittyAction.Stalk&&interruptedHunt.RoutineLength>=5,
    "A hunt episode does not start with a committed multi-stage stalking goal.");
var interruptions=interruptedHunt.InterruptedRoutineCount;
for(var i=0;i<14;i++) { interruptedHunt.Pester(); interruptedHunt.Tick(.1,new(1,0,1,0,1)); }
for(var i=0;i<30&&interruptedHunt.Routine!=KittyRoutine.Recovery;i++) interruptedHunt.Tick(.125,new(1,0,1,0,1));
Check(interruptedHunt.Routine==KittyRoutine.Recovery&&interruptedHunt.InterruptedRoutineCount>interruptions,
    "Repeated pestering did not interrupt the current goal and start a space-seeking recovery episode.");
var recoveryActions=new HashSet<KittyAction>();
for(var i=0;i<100;i++) { recoveryActions.Add(interruptedHunt.Action); interruptedHunt.Tick(.125,default); }
Check(recoveryActions.Contains(KittyAction.Backoff)&&recoveryActions.Contains(KittyAction.Peek),
    "A startled kitty did not back away and cautiously look again.");

var missingPrey=FindHunt(12); var lostCount=missingPrey.InterruptedRoutineCount;
for(var i=0;i<60;i++) missingPrey.Tick(.125,default);
Check(missingPrey.InterruptedRoutineCount>lostCount&&missingPrey.Routine!=KittyRoutine.Hunt,
    "A hunt stayed committed to a cursor that disappeared.");
Check(missingPrey.Action!=KittyAction.Pounce&&missingPrey.Action!=KittyAction.ButtWiggle,
    "The kitty still aims or pounces at absent prey.");

var napper=new KittyMind(21);
for(var i=0;i<8000 && !(napper.Action is KittyAction.Sleep or KittyAction.Doze && napper.ActionAge>.5);i++) napper.Tick(.125,default);
Check(napper.Action is KittyAction.Sleep or KittyAction.Doze,"The kitty never entered a wakeable nap.");
napper.Pester(); napper.Tick(.125,new(.8,0,0,0,1));
Check(napper.Action is not (KittyAction.Sleep or KittyAction.Doze),"A gentle interaction did not wake a light or deep nap.");
Check(napper.Willingness is >=0 and <=1&&napper.Frustration is >=0 and <=1,
    "Motivation or frustration is not a bounded simulation state.");
Check(!string.IsNullOrWhiteSpace(napper.EpisodeDescription),"The current episode has no readable goal description.");

var one=new KittyMind(100); var two=new KittyMind(100);
for(var i=0;i<1200;i++) {
    var senses=new KittyStimulus(i%90<20?1:0,i%75<10?1:0,i%40<20?1:0,i%130<2?1:0,1);
    if(i%111==0) { one.Pester(); two.Pester(); }
    one.Tick(.125,senses); two.Tick(.125,senses);
    Check(one.Action==two.Action&&one.ActionAge==two.ActionAge&&one.Attention==two.Attention&&one.Energy==two.Energy
        &&one.Routine==two.Routine&&one.RoutineStage==two.RoutineStage&&one.Frustration==two.Frustration,
        "A seeded cognitive simulation was not reproducible.");
    Check(double.IsFinite(one.MemoryActivity)&&one.MemoryActivity>=0&&one.MemoryActivity<=1,"Normalized input context diagnostic became unbounded.");
}
var age=one.ActionAge;
one.Tick(double.NaN,new(double.NaN,double.PositiveInfinity,-5));
Check(one.ActionAge==age,"Invalid elapsed time changed cognition.");
one.Tick(3600,new(double.NaN,double.PositiveInfinity,-5));
Check(one.ActionAge<=.5||one.ActionAge-age<=.5,"Hidden time was replayed as a huge jump.");
Check(!string.IsNullOrWhiteSpace(one.Intention)&&one.Thought==one.Intention,"Cognitive intention summary is absent.");

Benchmark();

void Benchmark()
{
    Span<double> output = stackalloc double[KittyBrain.ActionCount];
    var perf = new KittyBrain(); for (var index = 0; index < 100; index++) perf.Predict(calm, output);
    var samples = new double[1000];
    var allocated = GC.GetAllocatedBytesForCurrentThread();
    var started = Stopwatch.GetTimestamp();
    for (var index = 0; index < samples.Length; index++)
    {
        var before = Stopwatch.GetTimestamp(); perf.Predict(calm, output);
        samples[index] = Stopwatch.GetElapsedTime(before).TotalMilliseconds;
    }
    var elapsed = Stopwatch.GetElapsedTime(started);
    var inferenceAllocation = GC.GetAllocatedBytesForCurrentThread() - allocated;
    Array.Sort(samples); var p95 = samples[(int)(samples.Length * .95)];
    Check(p95 < 25 && elapsed.TotalMilliseconds / samples.Length < 20, "Full-context Transformer inference exceeded its nominal5Hz CPU budget.");
    Check(inferenceAllocation < 1024, "Steady-state Transformer inference allocated per-token managed objects.");
    Console.WriteLine($"PASS {checks} kitty checks; {KittyBrain.ParameterCount} trained float32 parameters / {KittyBrain.ParameterCount * sizeof(float):N0} weight bytes; "
        + $"full32-token inference mean {elapsed.TotalMilliseconds / samples.Length:F3}ms, p95 {p95:F3}ms; {inferenceAllocation} allocated bytes / {samples.Length} calls.");
}

void CheckPredatoryPlay()
{
    var hunting=new KittyMind(12); var hunted=new HashSet<KittyAction>();
    var huntedTransitions=new HashSet<(KittyAction,KittyAction)>(); var last=hunting.Action;
    for(var i=0;i<3200;i++) {
        var phase=i%200;
        hunting.Tick(.125,phase<70 ? new(.85,.55,1,phase<4?1:0,1) : new(.4,0,1,0,1));
        hunted.Add(hunting.Action);
        if(hunting.Action!=last) { huntedTransitions.Add((last,hunting.Action)); last=hunting.Action; }
    }
    Check(hunted.Contains(KittyAction.Stalk)&&hunted.Contains(KittyAction.Pounce),"Predatory play did not include stalking and pouncing: "+string.Join(", ",hunted));
    Check(huntedTransitions.Contains((KittyAction.Stalk,KittyAction.ButtWiggle))
        && huntedTransitions.Contains((KittyAction.ButtWiggle,KittyAction.Pounce)),"Stalk, aim and pounce episode continuity is absent.");
    Check(hunted.Contains(KittyAction.Sniff)||hunted.Contains(KittyAction.Inspect),"Novel surfaces never triggered investigation.");
}
