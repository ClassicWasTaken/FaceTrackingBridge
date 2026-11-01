# Relay: behavior, context and a personal bond

Relay is an original software pet with simulated needs, attention, a small
Transformer behavior policy, individual temperament and local memories. It
does not understand language or have conscious thoughts. The model chooses
behavior and attention preferences; ordinary application code maintains needs,
physical constraints, routines, interaction handling and the animated body.

## A genuine small Transformer

The policy uses two causal pre-LayerNorm Transformer blocks, eight attention
heads, a model width of 256, 512-unit feed-forward layers and sinusoidal position
encoding. It has 1,070,888 trained float32 parameters and two output heads for
35 behaviors and five attention targets: the cursor, surroundings, itself,
rest, or remembered interests. It evaluates the last 32 normalized observations
at a nominal 5 Hz cadence, about 6.4 seconds of recent active experience.

The app runs inference in C# with reusable buffers and one bounded background
job. Recent context belongs to the UI thread, and results are accepted only
when complete, current and valid. Hiding, resetting or reloading discards old
results. A stalled job never creates a queue or blocks the interface. It needs no Python,
PyTorch, GPU, network account, model download or cloud service. The renderer
continues at display refresh and blends articulated springs independently.

Weights were genuinely trained during development with CPU PyTorch on original
synthetic household-cat sequences. Research informs the scenario design; books,
articles, videos and real cats' records are not copied into a training corpus.
The model learns to imitate this design teacher, not to reproduce a biological
cat brain. [Training recipe, provenance and limitations](RELAY-TRAINING.md) and
[measured training report](RELAY-TRAINING.json) describe the exact run, held-out
episodes, unseen event orderings and context tests.

## Context that reloads

When the app opens, it reads the saved companion profile without enabling
Relay. Before the first active decision, it restores personality, familiarity,
preferences, shared memories and any still-recent Transformer context. The
kitty itself still starts **off on every launch**.

The save contains up to 32 quantized, normalized observation summaries. These
describe simulated needs and broad interaction signals; they do not contain
cursor coordinates, screenshots, keystrokes, audio, chat messages or raw input
logs. Recent context older than 30 minutes is discarded so a stale poke or
startle does not become a fresh event. Lasting traits and shared memories remain.

Only enabling the pet starts a memory session. Disabled first launches do not
create a save. Active decisions and meaningful interactions are coalesced into
small atomic saves; hiding, pausing and quitting also flush pending memories.
The primary file is bounded to 16 KB at
`%LOCALAPPDATA%\BabbleFrameBridge\kitty-memory.json`. A bounded last-good backup
can recover a damaged primary save without writing during startup. A bounded
unreadable copy preserves an unknown or malformed profile before a new session
replaces it; an oversized unknown file is left untouched. Forget memories also
clears those recovery copies. Missing, malformed or inaccessible memory cannot
stop face tracking from opening.

## A bond that grows

Each new companion receives an individual temperament, with varying playfulness,
sociability and caution. Its identity and traits persist. Gentle contact, shared
play, calm handling and familiar returns influence bounded trust, affection,
action preferences and social willingness. These values feed the policy and
attention controller, so memory affects behavior rather than being a display
label. Positive contact takes active time; a burst of calls or clicks cannot
instantly manufacture a close bond.

The current core-memory vocabulary is:

- **First gentle pet:** a sustained positive introduction.
- **Calm handling:** being picked up and gently set down.
- **Playing together:** completing shared play activity.
- **Comfort after a fright:** settling after a startle with gentle contact.
- **Familiar reunions:** recognizing an established saved relationship.

Repeated meaningful experiences reinforce a deduplicated memory. The first
moment remains identifiable while its strength can grow. Storage is bounded,
and the current five semantic categories never become an unbounded diary.
Core memories and temperament affect modeled comfort and preferences. Poking
can create temporary annoyance and lower willingness, but space and calm contact
allow recovery; the pet is not permanently punished or distressed by absence.

The million-parameter backbone remains fixed during use. A separate 40-value
personal output layer really does learn from calm, positive interaction using
bounded cross-entropy gradient updates. It changes behavior/attention
probabilities without retraining the large backbone. Updates are rate-limited,
require current neural predictions, and are saved with the companion profile.
A single poke is not treated as a negative preference label. Both this small
learning layer and the explicit bond, preferences and memories grow locally.
**Settings > Relay's memory > Forget memories** resets only this companion's
saved bond, temperament, preferences, learned reaction weights, motion practice
and recent context. Tracking modules,
connection settings and other application data are untouched.

## Petting, poking and pickup

Move the pointer gently across Relay's drawn body without holding a mouse
button to pet it. A stationary hover is not a pet; rapid waving is play
stimulation. A short click is a poke. Press and drag to pick it up, then release
to set it down. The legs and tail dangle, and a nearby usable ledge may catch
the drop. Escape, focus loss, hiding or lost capture cancels safely without
activating controls beneath the cat or inventing a positive handling memory.

Gentle strokes can encourage relaxed poses and silent purr bubbles. Repeated
pokes can make it grumpy; it also looks away, investigates its environment,
grooms, rests or plays independently. It does not constantly watch the cursor.
Hunting, settling, recovery, grooming, exploration and play unfold in connected
episodes rather than arbitrary pose changes. An eligibility controller keeps
sleep, movement and interaction choices physically appropriate.

Panel climbing and perching use the actual clipped edges of visible cards.
Changing tabs, scrolling and resizing refresh those affordances. Physics and
placement remain separate from the model's learned behavior scores. Relay
never clicks real controls, changes connection settings or moves the system
cursor. Visual swats and panel wobble are animation only.

Relay makes **no sound** and speaks no words. Meows, chirps, yawns, purrs and
other cat noises appear as brief click-through text bubbles. It pauses while
hidden or in the tray, and respects Windows reduced-motion preferences.

## Discovering and composing motion

Relay starts with hopping and a stretch/wave movement. Bounded practice and
positive feedback let it discover rollovers, bounding runs, running hops and
tumbles. A procedural motion grammar combines trajectories and varies tempo,
height, roll timing, stride and tail motion. Learned per-skill preferences,
mastery and motion prototypes persist between launches.

The neural mind supplies the current activity, energy, temperament and social
context; the motion controller chooses compatible experiments. Completing an
attempt provides small practice updates. Gentle petting during a movement
reinforces that candidate's prototype within a bounded reward budget. This is
local procedural skill learning, not arbitrary code generation or an unlimited
new animation vocabulary. It can generate varied motions from six known kinds.

Motions stay within finite geometric limits. Pickup, climbing, perching, hiding,
reset and insufficient space cancel a free-ground experiment. Reduced-motion
mode keeps interaction and personal learning available without animated hops
or spins. The cat never operates real interface controls.

## Behavior research and its scope

These sources inform original scenarios and animation choices, not a claim of
scientific validation for this digital pet:

| Source | Design influence |
| --- | --- |
| [AAFP/ISFM Environmental Needs Guidelines](https://catvets.com/resource/aafp-isfm-environmental-needs-guidelines/) | Safe resting places, elevated retreat, exploration, predatory play, and predictable positive contact. |
| [AAHA: Behavior and Environmental Needs - Kittens](https://www.aaha.org/resources/2021-aaha-aafp-feline-life-stage-guidelines/behavior-and-environmental-needs-kittens/) | Connected watching, stalking, swatting, chasing and pouncing; gradual gentle handling. |
| [Hall, Bradshaw and Robinson, 2002](https://www.sciencedirect.com/science/article/abs/pii/S0168159102001533) | Familiar stimuli can lose novelty while changed stimuli renew play interest. |
| [Humphrey et al., 2020](https://www.nature.com/articles/s41598-020-73426-0) | Relaxed slow-blink interactions. This study does not validate Relay's model. |
| [FelineVMA: Training Techniques for Addressing Undesirable Behaviors](https://catvets.com/resource/training-techniques-for-addressing-undesirable-behaviors/) | Predictable interactions, play opportunities and recovery rather than persistent aversive responses. |
| [AAHA on kneading](https://www.aaha.org/resources/is-my-cats-kneading-normal/) | Alternating paw motions before loafing and resting. |
| [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer architecture background; Relay is an original small implementation and training recipe. |

The exact timing, bond scale, core-memory categories, preferences and emotional
state variables are software design decisions. No third-party pretrained
weights or mascot images are bundled. Original artwork, policy weights and
synthetic training code use the project's Apache 2.0 license.

[Policy and context tests](../tests/Kitty.Tests/README.md) -
[Memory and gesture checks](../tests/Kitty.Memory.Tests/README.md) -
[Panel and pickup geometry](../tests/Kitty.Panel.Tests/README.md)
