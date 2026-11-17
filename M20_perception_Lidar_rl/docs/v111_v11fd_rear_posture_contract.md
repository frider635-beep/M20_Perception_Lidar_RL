# V11.1 V11-FD rear-posture contract

V11.1 inherits the saved V11-FD run at
`logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch`.
Its action, command, observation, event, termination, terrain, native reward,
network and PPO configuration must match that run. Compatibility fields added
to the source after the run remain at their saved-behavior defaults.

The only behavioral additions are bounded, descending-column rewards for rear
legs extending too far behind the body, rear knees folding beyond a free
region, and contacted rear wheels failing to roll with body motion, plus a
low-body cost after reaching level ground at the top or bottom of a stair
flight. These terms do not select a leg order or target tread. They add no
actor or critic observation and do not load a checkpoint or teacher.

The level-exit detector requires a completed stair sequence, an empty forward
riser scan, all four wheels in support, and a common-height wheel support plane
for a confirmation window. It never retains a particular tread. The cost uses
only root height above that support plane, so normal stair traversal stays free.

Rear-thigh alignment with the torso is deliberately absent. In particular,
V11.1 must not contain the later V13.4 rear-thigh parallel penalty, its weight,
geometry state, or metrics.

The task ID is `M20-Stair-V11.1-V11FD-RearPosture-v0`. Static tests and a real
Isaac step smoke establish the configuration boundary and runtime wiring; they
do not establish learned descent quality.
