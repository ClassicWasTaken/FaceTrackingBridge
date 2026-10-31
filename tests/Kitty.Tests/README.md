# Local Transformer kitty checks

Run `dotnet run --project tests/Kitty.Tests -c Release`.

Relay uses a real causal Transformer over the last 32 observation tokens:
22 normalized features per token, width 96, two pre-LayerNorm blocks, six
scaled dot-product attention heads of width 16, and a 192-unit GELU feed-forward
layer in each block. Fixed sinusoidal positional encoding represents order.
Separate softmax heads produce 35 behavior preferences and five attention modes:
cursor, environment, self, rest, and remembered interests. There are 155,848
trained float32 parameters (623,392 bytes), plus 12,288 bytes of positional data.
The app evaluates it directly in C# with reusable buffers and no inference
package, Python installation, GPU, network, or model download.

This is imitation learning over original synthetic temporal episodes. It does
not understand language or have conscious thoughts. All training is done during
development; petting changes bounded simulated familiarity, not neural weights.
The training script exports the model and Python golden predictions. Numerical
parity checks compare C# against those exported sequences, including variable
context lengths and a full rolling window. Synthetic teacher agreement does not
measure real-cat fidelity or prove intelligence.

The policy uses a nominal 5 Hz cadence during active simulated time. Controller
tick quantization can change individual intervals; it never replays missed
inferences in a catch-up burst. Its 32-token window covers about 6.4 seconds. Contact can interrupt a routine without
creating extra inference bursts. The first observation initializes the policy.
The window is bounded; observations older than its capacity are discarded.

Optional local memory contains individual playfulness, sociability and caution,
aggregate familiarity/preferences, a gradual bond, bounded interaction counts,
positive semantic episodes, and up to 704 quantized features for recent context.
Core episodes remember first gentle petting, calm pickup, shared play, comfort
after a startle, and a reunion. Repeated meaningful episodes can reinforce them
at bounded intervals; callbacks without elapsed active time cannot farm them.
These summaries influence preferences and existing model features, so they can
change later behavior. They do not constitute conscious or autobiographical
human memory. Familiarity grows without changing the trained model weights.
It contains no cursor coordinates, raw pointer log, or messages. Context loads
before the next policy decision; old context is discarded by the memory store
while longer-term familiarity survives. The neural weights remain fixed. The
companion still starts off each launch. Reset clears the Transformer sequence
and attention distribution. Store/gesture persistence checks live in their
separate regression suites.

Needs and explicit short-term traces retain play, rest, novelty and contact.
Trust changes with gentle contact versus pestering; habituation reduces the
importance of an unchanging cursor. Fatigue can override continued stimulation.
The network updates on a 200 ms context cadence and reconsiders direct contact,
while action choices and attention commitments last several seconds. Physical
eligibility, finite goal episodes and cooldowns turn action preferences into
coherent activities. The public routine/stage describes the current goal:

- Hunt: swat or chase → stalk → aim with a wiggle → pounce → inspect → wash a paw.
- Settle: circle → knead → loaf → doze → nap → stretch → yawn.
- Recovery: startle → back off → show annoyance → peek → sniff → relax.
- Grooming: general grooming → paw lick → face washing.
- Exploration: inspect → sniff → rub a familiar surface → look around → ear flick.
- Play: energetic play → tail chase or rolling → flop → wash or groom.

Activities check their prerequisites at every stage. Lost prey, fatigue, renewed
stimulation or repeated pestering can interrupt a goal. Recovery seeks space and
then reassesses; a light or deep nap can wake after direct interaction. The
completed/interrupted counts expose finite progress for diagnostics. Recent
action penalties and bounded same-session activity preferences reduce repeated
poses. These scalar preferences are simulated motivation, not online neural
training, and are persisted only in the optional bounded local memory.
Intention summaries describe the actual current simulated goal; they are not
generated reasoning transcripts. All original model, training code and synthetic
examples are covered by the project's Apache 2.0 license.

Behavioral grounding: [AAFP/ISFM environmental-needs guidelines](https://journals.sagepub.com/doi/full/10.1177/1098612x13477537),
[AAHA feline environmental/play guidance](https://www.aaha.org/resources/2021-aaha-aafp-feline-life-stage-guidelines/behavior-and-environmental-needs-kittens/),
and [the primary slow-blink communication experiment](https://www.nature.com/articles/s41598-020-73426-0).
These support broad activity motifs; the sources do not supply training data or
validate this software's model as a scientific simulation.

Checks cover Python/C# probability parity, true causal masking, different pasts
with the same present observation, rolling-window expiry, reset, bounded
quantized context export/restore, malformed context, nonfinite observations,
pet/poke familiarity, copied preferences, temperament identity across sessions,
positive core episodes, callback-spam limits, comforting recovery, learned
counterfactual history preferences, independent attention, habituation,
fatigue/recovery, goal interruption, absent prey, nap waking, deterministic
seeded simulations, and a measured full-context CPU/allocation budget.
Controller routine tests validate explicit behavior safety; they are distinct
from learned-policy history/parity tests.
