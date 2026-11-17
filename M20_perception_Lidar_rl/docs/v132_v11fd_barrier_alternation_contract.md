# M20 V13.2 V11-FD Barrier Alternation Contract

## Frozen baseline and scope

V13.2 inherits `DeeproboticsM20V13V11FDAlternationEnvCfg`.  V11-FD terrain,
commands, speed/progress/safety rewards and V13's generic safe single-riser
completion rewards remain unchanged.  V13.2 replaces only the six old V13
style terms with one independently learned style-return channel.

V13.2 never masks an action, gates generic stair credit, terminates an episode,
or enters an absorbing style-failure state.

## Ordered gait

The first unique FL or FR lift toward the first observed face defines B1 parity.
For seed side `s`, odd risers expect `s` and even risers expect `1-s`.  A skipped,
simultaneous or wrong-side event is charged once.  The target index is recovered
from the B1-anchored physical face position, so the next detected face cannot be
silently renamed B1 and the fixed parity remains recoverable after a mistake.

Historical-best target progress is paid once.  A relaxed log-barrier score
modulates expected-leg progress inside a lead-margin band.  Repeated-side,
bilateral-overlap, skip and overrun terms cannot collect positive progress.

## Last riser and platform

No-riser absence is not platform evidence.  Top-follow begins only after:

1. one front side physically completed the last observed face;
2. that leader remains supported beyond the face;
3. forward PGTT ray hits contain a finite, low-relief patch near the completed
   upper height, across the configured longitudinal span and lateral corridor;
4. no higher surface appears in that corridor; and
5. the evidence persists for the configured confirmation window.

The opposite side then targets that same upper plane, not a fabricated next
riser.  Its historical-best potential requires height, forward edge crossing
and support.  Stable follower support enters plateau-roll; unnecessary front
lift is penalised until a genuinely distinct later staircase re-arms B1.

The same transition handles one-riser and multi-riser stairs.

## Actor observation and deployment ABI

The actor is 68D:

```text
0:63   frozen V11 actor input
63     expected side: -1=FL, 0=unassigned, +1=FR
64:68  one-hot: APPROACH_B1, ALTERNATING_RISERS,
        TOP_PLATFORM_FOLLOW, PLATEAU_ROLL
```

These five values are finite-state outputs, not privileged terrain labels.
Deployment must run the same persistent ledger beside the existing PGTT
preprocessor:

- B1 and later face identity come from the same two-riser PGTT extraction plus
  odometry anchoring.  If a face is at yaw-frame `(D,Y)`, its world anchor is
  `p_base_xy + R_yaw * (D,Y)`; its integer index is
  `round(dot(p_face-p_B1, flight_axis)/step_width)+1`;
- wheel centres come from joint encoder forward kinematics and wheel bottom is
  `z_FK - wheel_radius`. Before the face, forward-moving normalized clearance
  is `(wheel_bottom-lower_plane)/(step_height+clearance_margin)`. The first
  unique side crossing the configured B1 onset seeds FL/FR parity. A bilateral
  onset leaves parity unassigned;
- contact-free geometric support means the wheel centre has passed the face by
  `wheel_radius + front_margin` and its bottom is within
  `upper_support_tolerance` of the PGTT upper plane for the configured number
  of consecutive control frames;
- top recognition consumes the existing local PGTT ray-hit grid and requires
  the positive same-height plane test above;
- reset clears the five-state ledger; a new distinct stair flight re-arms it.

The training implementation is `v132_deployable_phase_state`; it is isolated
from the contact-backed reward ledger and reads only LiDAR/PGTT, base pose and
joint-state/FK equivalents.  The deployment-side implementation must reproduce
these transitions exactly.  A model must not be promoted to hardware if the
deployed observation builder still emits the old 63D ABI, uses a contact/reward
event to make these five values, or substitutes `B1=B2=0` for positive platform
evidence.

Five extra scalars are not a mathematical requirement for alternation.  They
are the chosen memory ABI for this feed-forward V13.2 actor.  Removing them
requires retraining with an equivalent memory carrier (for example a recurrent
actor or an observation-history window); a 68D checkpoint cannot be deployed
through the old 63D input path.

The contact-backed reward ledger settles style progress only when its physical
riser index matches the deployable ledger's same-step settlement snapshot. A
temporary mismatch is fail-closed (zero style cashflow), is reported by
`style_target_gate_rate`, and can never reward or punish the leg opposite to
the actor's 5-D instruction. Generic V13/V11 task reward remains active during
such a mismatch.

## Dual-return PPO

The environment logs the sum of native and style reward, but PPO reads the exact
settled `v132_style` value from RewardManager's per-step ledger and subtracts it
to recover the native task reward.  Recomputing the term after `env.step()` is
forbidden because terminal environments may already have reset their state.
Separate critics and GAE returns are used for native task and style. Their
advantages are standardised independently and added with fixed equal weights;
the native task advantage is never divided down. A rollout containing no
non-zero style settlement contributes a
strictly zero style advantage, so random style-critic residuals cannot create a
phantom actor gradient.  Both critics and the optimizer are checkpointed;
V13.2 refuses checkpoints that do not contain the style critic.

## Required preflight

`scripts/tools/test_m20_v132_barrier_alternation.py` and the strict V13↔named-V11
snapshot audit must pass before launch. Runtime smoke must additionally show
actor 68D, action 16D, two distinct critic objects, a live `v132_style`
RewardManager term, the intended V13.2↔V13 config delta, and finite
observations/rewards.
One-step, multi-step, skipped-B1, simultaneous lift, wrong-side, platform follow
and post-platform lift rollouts remain required before model promotion.
