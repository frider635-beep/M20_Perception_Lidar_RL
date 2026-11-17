# V13 native task with critic-only reward state

This task inherits V13 without changing its rewards, terrain, commands, events,
terminations, curriculum, actor, action space, PPO settings or network widths.
The actor remains the deployable native 63D observation and produces 16 actions.

The existing V13 critic already observes clean privileged terrain information.
This variant appends six values that resolve hidden state in V13's historical
soft-alternation reward: expected-FL, expected-FR, sequence-started, active
reward target, expected-leg current potential and expected-leg historical-best
potential. These values are simulator-only and never enter the actor or export.

Runtime acceptance requires actor 63D, critic 574D and action 16D, plus an exact
comparison of every behavioral configuration field against native V13.
