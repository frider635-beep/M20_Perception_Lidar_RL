# 2026-11-17 迁移更新

默认模型改为同级训练工程的 `interview_prep/model_35300.pt`，已补齐其原始 `params/env.yaml` 与 `agent.yaml`。启动脚本使用当前环境 Python，避免依赖固定 Conda 安装目录。训练记录和其它 checkpoint 不需要随验证程序迁移。

在该保留模型上执行 `test_contract.py`：14 项通过，覆盖原始 actor/感知对照、配置选择、按键、起立复位与到顶继续运行。到顶测试使用 14 秒物理仿真；旧 12 秒限制不足以让该基线完整触发终点。配置迁移测试直接验证打包的 params 与显式指定配置，不再依赖未分发的旧训练日志。

同次训练工程的修复 CPU 测试 39 项通过，迁移预检查通过；仅使用 CPU，未启动新训练。以上不是改进模型行为验收。下文是旧版本历史记录。

# V13.5 MuJoCo validation record

Scope: user-selected model_73299.pt and stairs_ascent_walls.xml. New isolated Git worktree; no edits to original training checkout.

Resources before execution (2026-11-16, Asia/Shanghai): MemAvailable 54 GiB, swap unused, load 0.29/0.37/0.34, GPU 621/49140 MiB occupied by desktop; 817 GiB free on workspace disk. CPU inference, batch 1, torch/BLAS threads 1, one simulation at a time. The actor has roughly 0.2 million FP32 parameters (<1 MiB); checkpoint <10 MiB plus Torch native libraries, robot meshes (26 MiB) and MuJoCo runtime: reserve 4 GiB host RAM. No CUDA model allocation. Short smoke was bounded to three simulated seconds; further validation runs use a transient user unit MemoryMax=4G, CPUQuota=200%, RuntimeMaxSec=180. User cgroup memory.max is max; no local resguard service found in the searched user/system unit directories.

Source provenance: MIGRATION.json records exact upstream files and hashes. Terrain loader/reset/keyboard modules and MJCF meshes are migrated, with no changes to the source deployment tree.

Workflow: inspect saved ABI -> differential tests -> flat and staircase closed-loop -> optional video/viewer -> commit. Failure recovery: retain original checkpoint/config/assets; output paths never overwrite an existing run. No hardware/ROS connections.

Hub integration skipped: this machine has no live ~/claude-knowledge or kb, no hub Git remote, and the archived knowledge copy contains neither project-space executable nor project-space-lifecycle.md. The Git post-checkout startup hook points to missing /home/ubuntu/claude-knowledge/bin/project_startup.py; worktree/branch creation succeeded but that hook failed. This local record is not presented as a Hub project Space. Origin fetch returned HTTP 403; GitHub returned Repository not found. No push to GitHub.

## Executed results

Runtime: Python 3.11, MuJoCo 3.3.7, Torch 2.7.0+cu128 (CPU placement), NumPy 1.26.0, PyYAML 6.0.2; video imageio 2.37.0/imageio-ffmpeg 0.6.0. Reference tests used RSL-RL 5.0.1 and pytest 9.0.3.

- Eight contract tests passed: 256 randomly generated signed-edge/diagonal/missing-ray maps matched the frozen training extractor; actor matched installed RSL-RL on 128 random inputs; observation order/scaling/wheel masking; target promotion and command gating; temporal confirmation/hold/delay/reset; named joint/actuator roundtrip, PD and actual terrain ray extraction; missing config rejection; incompatible actor rejection.
- Default ascent: goal_reached in 10.76 simulated seconds, 538 control steps, no detected fall or bypass. This uses the unchanged user-provided terrain template. Evidence: runs/ascent_73299 and identical rendered run runs/ascent_video_v2_73299. Final base [4.122393, 0.078245, 2.475497].
- Descent preset: goal_reached in 13.66 simulated seconds, 683 control steps, no detected fall or bypass. Evidence: runs/descent_73299. Final base [4.612480, -0.100135, 0.553641]. The descent template has its own saved parameters; it is not treated as the same terrain as ascent.
- Flat control: eight simulated seconds with no detected fall; no false active stair feature frames. Evidence: runs/flat_73299. Duration limit reached before its distant geometric finish.
- Native desktop viewer/keyboard mode opened, executed 100 control steps and exited normally after two simulated seconds. Evidence: runs/viewer_smoke. Manual motion-key and reset/focus-loss interaction was not exercised by an operator.
- EGL MP4 rendering succeeded; the six-second frame was decoded and visually inspected. Video: runs/ascent_73299.mp4.
- Measured process peak RSS with /usr/bin/time: ascent 928872 KiB; flat 930408 KiB; descent 928336 KiB; EGL video 981208 KiB. The transient unit's displayed memory peak was implausibly small, so it is not used as a measurement. Four-GiB MemoryMax and two-core CPUQuota were applied to subsequent validation.

Earlier failed attempts were corrected: unrelated auto-loaded pytest typeguard plugin required disabling plugin autoload; video framebuffer needed explicit offscreen dimensions before constructing Renderer. No installed dependency was changed.

## Boundaries

One deterministic seed and fixed commands were exercised. Strict per-riser alternation, physical clearance/impact quality, long-term flat behavior, randomization/noisy robustness and hardware acceptance were not measured. Goal-reached is the documented geometric/support criterion, not a claim of those properties. The virtual down-looking grid is a simulator terrain sensor; it does not reproduce real LiDAR occlusion, mapping or transport. No training or robot service was launched or modified.

Git branch: agent/codex/v135-mujoco-validation. Local commit is the delivery; Hub push/merge is unavailable because no hub remote is configured. No GitHub write is authorized or performed.

## Relocated checkpoint configuration fix (2026-11-16)

Classification: local configuration lookup failure. A standalone checkpoint cannot supply the saved observation/action semantics. The interview_prep/model_35300.pt SHA-256 is 4aefe17ddc279ee809ff6257862ba4e28bb1b19ad48b11a5948b116256d61f8b, identical to M20-V11.1-V11FD-RearPosture/2026-09-26_12-13-33_v111_native_ascent_from_v11fd29999_e8192_r24/model_35300.pt. The same-named V13.5 checkpoint has a different hash and must not supply this model's configuration.

Configuration resolution now uses adjacent params/, an explicit --config-dir, or the complete SHA-256 of a same-named checkpoint in the known local training-log layout. Partial adjacent configs, missing matches and conflicting matching configs fail with instructions. The resolved env.yaml is passed through to the joint-semantics reader; both configuration paths/hashes and match provenance are recorded in manifest.json. No training artifacts were copied or modified. For other machines, preserve both original YAML files with the checkpoint or supply their directory explicitly; network shape alone is not evidence that training semantics match.

Resources: previous measured test peak ~1 GiB (above); before execution MemAvailable 52 GiB, swap unused, load 0.33/0.37/0.36, disk free 817 GiB. One CPU-only process at a time, batch 1, Torch/BLAS threads 1; transient user units MemoryMax=4G, CPUQuota=200%, RuntimeMaxSec=180 (viewer smoke 60). No GPU model allocation. Existing executable-bit changes throughout the validation tree were left unchanged and excluded from this fix's commit.

Executed: 14 contract/regression tests passed in 6.57 s, peak RSS 1016172 KiB. New coverage rejects a different same-named checkpoint and ambiguous/incomplete configs, compares the real copied V11.1 actor and semantics with the original run, and exercises explicit config selection. The first test attempt had an incorrect test fixture parent path (13 passed, one failed); corrected and reran the full suite successfully.

Production launcher with the copied checkpoint ran 12 simulated seconds / 600 control steps on the ascent terrain: no detected fall or bypass; reached base x=3.7052, z=2.4642. It did not meet the geometric finish criterion within the selected duration. Peak RSS 929928 KiB; evidence runs/v111_35300_relocated. This is a load and rollout check, not strict gait acceptance. Native viewer/keyboard startup was also checked for two seconds with an explicit duration; manual key operation on this checkpoint and hardware validation were skipped. The existing default V13.5 tests continued to pass.

## Numeric keyboard update (2026-11-16)

User-requested bindings: 8/5 forward/backward, 4/6 left/right translation, 7/9 left/right yaw. All movement sources share key_bindings.py; top-row and numeric-keypad GLFW events, terminal characters and X11 held-key polling are connected. Legacy WASDQE movement is disabled; mode/reset/quit keys are retained.

Executed 18 controller routing probes (six commands across top-row, keypad and terminal), checked ramped target velocity and immediate clear-motion stop, rejected all old movement keys, retained Z/C/R/Backspace/Escape events, and verified X11 top-row/keypad keycode mappings plus held/released bitmap decoding. All passed. No synthetic input was sent to the desktop. MemAvailable 54 GiB with no swap; a single short CPU-only probe loaded no model or simulator and completed in 0.063 seconds. Full physical rollout tests skipped for this update because only keyboard routing and UI text changed. An existing viewer must be restarted to load the updated modules.

## Interactive lifecycle update resource plan (2026-11-16)

Before execution: MemAvailable 53 GiB, swap unused, load 0.27/0.49/0.43, workspace disk free 817 GiB. Prior measured simulator+Torch process peak was below 1 GiB. Budget one CPU batch-1 process with 4-GiB MemoryMax, CPUQuota=200%, RuntimeMaxSec=180 and single-thread BLAS/Torch. The new checks execute real MuJoCo stand-up/reset trajectories and the production loop with deterministic keyboard events; they do not send input to the desktop or control hardware.

Lifecycle validation results: all 11 tests passed in 5.72 seconds; process peak RSS 1010504 KiB. Three new tests use real physics: full prone -> Z stand -> ready -> C policy -> R stationary snapshot -> Backspace prone; R also works before any completed stand. Production-loop event injection additionally checks early C waits, no inference/motion while prone/standing/ready, cleared action history on R, and no automatic policy re-entry after Backspace. Default-goal-continuation test reached the original ascent finish and still executed the complete 600 control steps/12 simulated seconds, ending by explicit duration rather than goal. No operator keyboard events were injected into the desktop.

Viewer default now has no duration limit. Long sessions save bounded trajectory chunks; finish events are latched and logged without stopping. --stop-at-goal is opt-in for batch evaluation. Viewer falls leave the window open for R/Backspace; headless fall behavior remains explicit in its manifest. R restores the most recent completed stand world pose with zero velocities, forces and command/policy/perception history; Backspace restores original spawn prone pose. Original per-riser gait/noise/hardware acceptance boundaries still apply.

Native viewer smoke: opened for two simulated seconds, stayed prone without an operator Z/C event, and exited normally by the explicitly supplied duration. Evidence: runs/viewer_prone_lifecycle. Long-session recorder check: 83 simulated seconds of zero-command inference on flat ground, 4150 control steps; persisted 4096 rows in trajectory_00000.npz and 54 rows in trajectory.npz, with no dropped rows and no detected fall. Peak RSS was 931328 KiB. Evidence: runs/session_chunk_check. These checks do not claim manual keyboard interaction, exact zero physical drift or long-distance walking performance.
