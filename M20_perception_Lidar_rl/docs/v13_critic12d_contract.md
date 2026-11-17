# V13 native actor ABI with balanced stair learning

This task keeps native V13's deployable 63D actor, 16D action, terrain,
events, terminations, PPO optimizer and action limits. The actor receives no
expected-side, gait-phase or desired-action input.

The critic retains native V13's 568D clean state, adds the six-value ascent
reward lifecycle used by the Critic6D variant, and adds six descent values:
descending course active, live negative edge, edge longitudinal distance,
drop height, maximum rearward wheel extension from the episode-start neutral
reach, and rear-wheel contact fraction. Course, extension and contact remain
visible after the edge leaves the forward scanner. They describe current
physics and do not prescribe a leg order or retain a stair identity.

The measured rollout-32 experiment learned velocity faster while producing
roughly one tenth of native V13's stair completions. This task therefore
restores the native 24-step rollout. On ascending stair columns only, commands
are sampled in a moderate range and speed-deficit pressure is reduced. Linear
and yaw tracking rewards are paid only in proportion to real forward progress
along the fixed world route, heading alignment and lateral corridor quality.
They remain native outside ascending stair columns. World-forward progress,
heading and lateral route terms remain active, so stopping, turning away or
moving sideways cannot retain the positive ascent tracking reward.

Descending terrain has two bounded, course-wide physical costs. Rear wheels
may move freely within an episode-start reach allowance; only additional
rearward extension is penalized. Contacted rear wheels also receive a
rolling-consistency cost. Neither term locks a tread, specifies a joint angle,
or imposes left/right order.

Read-only metrics report course and live-edge coverage, edge distance, drop
height, maximum rearward extension and rear-wheel contact fraction. They do
not gate terrain movement.

Runtime acceptance is actor 63D, critic 580D and action 16D, followed by one
successful PPO update from random initialization. Training uses native V13's
24-step rollout and a 20,000-iteration from-scratch limit.
