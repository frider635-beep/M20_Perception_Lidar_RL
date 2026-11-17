# V13.4 native ascent and natural descent contract

V13.4 directly inherits the original V13 task. The actor remains the native
63D observation and 16D action policy. Commands, terrain distribution, native
velocity/forward progress, physical landing, safety terms and PPO update all
remain V13 behavior. The actor, critic, optimizer and iteration start fresh.
The reviewed V13 `model_5000.pt` is loaded into a separate frozen teacher.

On ascending stairs, native speed and route rewards remain fully active before
the first support. Training-side geometry locks the first observed ascending
face in world space as B1. Either FL or FR may freely seed the chain, but only
by stable support on that face; support first established on B2 cannot be
renamed as B1. Contact accounting then recognizes only an opposite-side support
on the immediately adjacent 0.30 m tread. A correct adjacent transition pays
the configured pair dividend; the third support in FL/FR/FL or FR/FL/FR pays an
additional chain dividend. Before stable support, historical-best progress of
the expected leg toward that exact adjacent tread supplies a smaller bounded
positive teaching signal. Repeat, skip, simultaneous motion and failed
attempts receive no added style cost or clawback, so avoiding the stairs cannot
improve the V13.4-specific return.

The learned Gaussian policy remains the main online exploration source. V13.4
does not choose the first side, enforce a noise floor or change joint limits.
Flat, descent and platform behavior update from the first iteration with the
reviewed learning rate and V13's original entropy coefficient. A frozen teacher
can execute sparse, coherent eight-step rollout segments only when the terrain
is ascending, the forward command is positive and a positive riser remains
visible. The probability of starting a segment decays during training. Leaving
that eligibility region ends the segment immediately, so descending states,
flat terrain, non-forward commands and the riser-free top never execute teacher
actions.

Teacher-executed transitions remain in the rollout and therefore contribute
environment rewards, return targets and critic regression. They do not enter
the Actor PPO surrogate: the runner retains the student's sampled action and
log probability for bookkeeping, marks the executed teacher transition, and
zeros its advantage before the PPO update. This avoids treating an off-policy
teacher action as though the fresh student sampled it. Student-controlled steps
before and after the segment keep their GAE credit from adjacent progress and
chain rewards, allowing those rewards to retain useful approach and continuation
behavior. There is no teacher KL, behavior-cloning loss, actor initialization or
parameter interpolation. Metrics expose teacher action occupancy, eligible-state
occupancy, burst count and the number of Actor samples masked each update.

The verified V13 `model_5000.pt` trajectory is both the behavioral acceptance
reference and the ascent-only rollout teacher. It does not initialize the
student, critic, optimizer, iteration or environment state. This run trains a
fresh student while exposing PPO to a small number of useful ascent states.

Descending stairs have no expected side, leg order, target-tread lock or rear
follow state machine. The original V13/V11-FD velocity, stability and physical
descent rewards remain active. Three bounded posture costs act over the whole
forward-commanded descending course: one rejects excessive rearward wheel
reach, one rejects excessive rear-knee folding, and one rejects only the
extreme in which a rear thigh becomes nearly parallel to the torso. Each has a
free region, so ordinary rear-leg flexion and extension remain available.
Rear-wheel rolling consistency remains soft and symmetric.

After the final riser, reward state retains the last face long enough for rear
crossing and stable support to confirm the platform. When the scanner has no
visible riser and a front wheel supports the last completed tread, a bounded
low-body-clearance cost begins before rear support confirms the platform. This
rejects the sprawled/crawling posture without prescribing joint targets.

The critic receives twelve training-only values: six for the contact-backed
ascent chain and six for current descent geometry/rear motion. The actor never
receives them. There is one standard scalar V(s) critic; V13.4 does not use the
previous terrain-separated three-critic runner, priority-gradient projection,
delayed reward channels or chain-triggered exploration.

Runtime acceptance requires a 63D actor, 580D critic, 16D action, exact teacher
equality with the reviewed checkpoint, a fresh student, critic and optimizer,
finite environment steps and one complete 24-step PPO update. The update must
execute ascent-only teacher actions, retain their critic targets, and mask every
teacher-executed transition from the Actor PPO surrogate.
Behavioral acceptance still requires frozen-checkpoint rollouts and sim2sim:
ascent must enter B1 and increase adjacent/chain events, top traversal must keep
normal body clearance, and descent must avoid rearward trailing, knee-wheel
folding and the thigh/torso-parallel extreme without a prescribed gait.
