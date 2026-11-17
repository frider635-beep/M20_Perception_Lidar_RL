# V13.5 ascent detector repair — 2026-09-20

Scope: existing task; descent, flat recovery, Actor63 MLP, Critic574, Action16, PPO8192×24 and reward weights unchanged.

## Evidence
- Neutral FL/FR lateral ±0.226576m exceeded the scanner matching tolerance ±0.15m incorrectly reused for wheel landing/potential. Realistic wheel fixture failed before repair and passes after.
- Physical tread corridor now uses terrain-origin transverse position, locked cardinal direction, per-riser inner-ring span minus wheel radius. Scanner matching tolerance unchanged.
- Same model12800 deterministic128env×600step audit: valid14→20, ordered0→4, max target2→3. Model weights unchanged. This proves detector impact, not learned policy improvement.
- Delayed native landing after strict settlement formerly counted as duplicate. Suppressed only for same physical edge, same foot maintaining uninterrupted contact; new swing re-arms duplicate detection.
- Resume restores control-step schedule, avoiding renewed early descent/reset anneal.

## Validation
All six V135 CPU contract suites and compile pass. Final real Isaac16env×8step + one MLP PPO update PASS: /tmp/v135_lateral_final_smoke.log. Snapshot: .codex-scratch/v135_lateral_release.

## Limitations
No claim that reward ordering or long-term strict alternation is solved. Pool edge-height matching retains ±0.15m lane tolerance; lateral drift may still prevent target-height acquisition. Monitor this before attributing remaining failure to PPO/reward. No new reward or observation added.

Training: continuation from latest intact checkpoint, separate output directory, target total30000 iterations; exact source hashes and weights in new run manifest.

Service: `m20-v135-ascent-repair-20260920.service`. Source checkpoint: old run `model_13100.pt` (optimizer included). Additional16900 updates → total30000; old service stopped with SIGINT. New run prefix2026-09-20_22-57-50.

Runtime verified: logs/rsl_rl/M20-V13.5-Final-FromScratch/2026-09-20_22-57-50_v135_ascent_physical_tread_first_swing_resume13100_e8192_r24; ABI63/574/16,8192×24. Reward weights exactly equal prior run; source manifest hashes match current files. Initial update snapshot:51 full ascent episodes,B1=33,B2=8,B3=2,full chain=0; insufficient for learning/retention verdict.

## User correction: total target43100
User requested30000 additional iterations starting at13100. Existing segment continues uninterrupted to30000; `m20-v135-continue-to43100.service` waits for successful completion and resumes final model29999 with13101 iterations (installed runner uses exclusive upper bound) to target43100. Failure/interruption/missing final checkpoint blocks continuation. Both services must be stopped to cancel the full plan. This is scheduled continuation, not yet executed.

## Restart after missing log directory
Previous service failed23:01 with TensorBoard FileNotFoundError; cause of directory disappearance unknown. Pending continuation also failed closed. Restarted directly from intact model13100 with30000 additional updates (target43100), service m20-v135-repair-restart-43100, PID2497486. Actual updates13105→13109 observed. New run:logs/rsl_rl/M20-V13.5-Final-FromScratch/2026-09-20_23-04-31_v135_ascent_repair_resume13100_plus30000_e8192_r24. Manifest/source unchanged from validated release; checkpoint and TensorBoard event writes verified.
