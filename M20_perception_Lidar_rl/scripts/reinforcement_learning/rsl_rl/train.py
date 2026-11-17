# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.", allow_abbrev=False)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--initial-checkpoint", type=str, default=None,
                    help="EntryFlatRepair: initialize from an exact checkpoint path with fresh Adam state")
parser.add_argument("--guarded-total-updates", type=int, default=None,
                    help="Recovery only: total additional updates with periodic behavior reports; valid behavior failures continue")
parser.add_argument("--guard-baseline", type=str, default=None,
                    help="Results JSON from the original 35300 on this host, including keyboard cases")
parser.add_argument("--guard-output", type=str, default=None,
                    help="New session directory for status and periodic independent evaluation")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Stop oversized or unbounded repair stages before constructing Isaac Sim.
from repair_workstation_limits import TASKS as REPAIR_TASKS, validate as validate_repair_limits
if args_cli.task in REPAIR_TASKS:
    args_cli.num_envs, args_cli.max_iterations = validate_repair_limits(
        args_cli.num_envs, args_cli.max_iterations, distributed=args_cli.distributed)
if args_cli.guarded_total_updates is not None:
    from guarded_recovery import validate_session
    validate_session(args_cli.task, args_cli.guarded_total_updates, args_cli.max_iterations,
                     args_cli.guard_baseline, args_cli.guard_output)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
if args_cli.headless and args_cli.task in REPAIR_TASKS:
    # Isaac installs SIGTERM -> app.close() and SIGINT -> native shutdown.
    # Replace those handlers before training so user stops can unwind safely.
    from guarded_recovery import TrainingInterrupted, install_interrupt_handlers
    install_interrupt_handlers()

# suppress noisy omni.usd warnings (e.g. unresolved visual prim references)
import carb
carb.logging.acquire_logging().set_level_threshold_for_source(
    "omni.usd", carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR
)

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
try:
    from rsl_rl.utils import resolve_callable
except ImportError:
    def resolve_callable(path):
        """Compatibility for the official RSL-RL 3.x runtime used on M20."""
        module_name, attr_name = path.rsplit(":", 1) if ":" in path else path.rsplit(".", 1)
        return getattr(__import__(module_name, fromlist=[attr_name]), attr_name)

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:
    def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version):
        # RSL-RL 3.x consumes the legacy `policy` configuration directly.
        return agent_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401
from rl_training.envs.amp_locomotion_env import AmpLocomotionEnv, AmpRslRlVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    entry_flat_repair = args_cli.task in REPAIR_TASKS
    if entry_flat_repair and args_cli.num_envs is None:
        raise ValueError("EntryFlatRepair requires explicit --num_envs; use its workstation-limited launcher")
    if args_cli.initial_checkpoint is not None and args_cli.resume:
        raise ValueError("Choose --initial-checkpoint for a new objective, or --resume for continuation")
    fresh_entry_flat = entry_flat_repair and args_cli.initial_checkpoint is not None
    if args_cli.initial_checkpoint is not None and not entry_flat_repair:
        raise ValueError("--initial-checkpoint is only supported by EntryFlatRepair")
    if fresh_entry_flat:
        agent_cfg.resume = True
    if entry_flat_repair and not agent_cfg.resume:
        raise ValueError("EntryFlatRepair requires --initial-checkpoint or --resume")
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    runner_class_name = getattr(agent_cfg, "class_name", "OnPolicyRunner")
    is_custom_runner = runner_class_name not in ("OnPolicyRunner", "DistillationRunner")
    # Custom V10/V11/V12 bridges intentionally consume the legacy ``policy``
    # block and convert it to the RSL-RL 5.x actor/critic ABI themselves.
    # Applying Isaac Lab's generic migration first clears that block, leaving
    # the custom bridge with an empty/default policy and causing a misleading
    # policy-configuration failure.  Built-in runners still use the official
    # migration path.
    if not is_custom_runner:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S_%f" if args_cli.guarded_total_updates is not None else "%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    if args_cli.task in REPAIR_TASKS and os.environ.get('M20_REPAIR_RUN_RECEIPT'):
        from pathlib import Path
        receipt = Path(os.environ['M20_REPAIR_RUN_RECEIPT'])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(log_dir+'\n')

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    _algo_class_name = getattr(getattr(agent_cfg, "algorithm", None), "class_name", None)
    if agent_cfg.resume or _algo_class_name == "Distillation":
        if fresh_entry_flat:
            resume_path = os.path.realpath(args_cli.initial_checkpoint)
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(resume_path)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    if isinstance(env.unwrapped, AmpLocomotionEnv):
        env = AmpRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    if version.parse(installed_version) < version.parse("4.0.0"):
        # The current IsaacLab config schema contains a future `beta` field
        # that is not accepted by the official RSL-RL 3.x PPO constructor.
        train_cfg.get("algorithm", {}).pop("beta", None)
    runner_class = resolve_callable(runner_class_name) if is_custom_runner else OnPolicyRunner
    runner = runner_class(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if fresh_entry_flat:
            # A changed objective needs fresh Adam moments and the configured
            # fine-tuning LR, while preserving the source iteration/schedule.
            checkpoint_infos = runner.load(resume_path, load_cfg={"actor": True, "critic": True,
                "optimizer": False, "iteration": True, "rnd": False})
            if runner.alg.optimizer.state:
                raise RuntimeError("EntryFlatRepair expected fresh optimizer state")
            print(f"V111_REPAIR_LR={runner.alg.learning_rate}", flush=True)
        else:
            checkpoint_infos = runner.load(resume_path)
        if args_cli.task in (
            "M20-Stair-V13.5-V11FD-NativeAlternation-v0",
            "M20-Stair-V11.2-VariableTread-v0",
            "M20-Stair-V11.1-EntryFlatRepair-v0",
            "M20-Stair-V11.1-AdjacentRepair-v0",
            "M20-Stair-V11.1-AscentRecovery-v0",
        ):
            # Resume the training schedule too: restarting it would re-enable
            # early reset/command distributions and soften descent rewards.
            steps_per_env = int(agent_cfg.num_steps_per_env)
            if args_cli.task == 'M20-Stair-V11.1-AscentRecovery-v0' and fresh_entry_flat:
                from pathlib import Path
                import yaml
                source_agent = yaml.safe_load((Path(resume_path).parent/'params/agent.yaml').read_text())
                steps_per_env = int(source_agent['num_steps_per_env'])
            env.unwrapped.common_step_counter = int((checkpoint_infos or {}).get(
                'm20_common_step_counter', int(runner.current_learning_iteration)*steps_per_env))
            env.reset()
            prefix = "V111_REPAIR" if entry_flat_repair else (
                "V112" if args_cli.task == "M20-Stair-V11.2-VariableTread-v0" else "V135")
            print(f"{prefix}_RESUME_CONTROL_STEP={env.unwrapped.common_step_counter}", flush=True)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # V13.5 captures the exact source/config before sampling and writes a report
    # even after a user interrupt. Other tasks keep the existing runner path.
    v135_report = args_cli.task == "M20-Stair-V13.5-V11FD-NativeAlternation-v0"
    if v135_report:
        from rl_training.v135_training_report import write_manifest, build_report
        write_manifest(log_dir, env, runner, agent_cfg)
    run_status = "completed"
    try:
        if args_cli.guarded_total_updates is not None:
            from guarded_recovery import learn_guarded
            learn_guarded(runner, total_updates=args_cli.guarded_total_updates,
                          segment_updates=agent_cfg.max_iterations,
                          baseline_path=args_cli.guard_baseline,
                          output_path=args_cli.guard_output, log_dir=log_dir)
        else:
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except BaseException as error:
        run_status = type(error).__name__
        raise
    finally:
        if v135_report:
            try:
                logger = getattr(runner, "logger", runner)
                writer = getattr(logger, "writer", getattr(runner, "writer", None))
                if writer is not None:
                    writer.flush()
                print(f"[V135 REPORT] {build_report(log_dir, run_status)}", flush=True)
            except Exception as report_error:
                from pathlib import Path
                Path(log_dir, "V135_REPORT_ERROR.txt").write_text(repr(report_error))
                print(f"[V135 REPORT FAILED] {report_error!r}", flush=True)
        writer = getattr(getattr(runner, 'logger', None), 'writer', None)
        if writer is not None:
            writer.flush()
            writer.close()
        env.close()


if __name__ == "__main__":
    if args_cli.headless and args_cli.task in REPAIR_TASKS:
        # This host's Isaac Sim 5.1 Kit faults in native framework teardown,
        # after training, checkpoint and environment cleanup have completed.
        # Do not invoke that teardown for this standalone headless process.
        # Preserve real failure exit codes; never turn a failed gate into success.
        try:
            main()
        except TrainingInterrupted as error:
            print(f'[M20] {error}; interrupted segment is not an accepted checkpoint.', flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(128 + error.signum)
        except BaseException:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        print('[M20] Training target finished; logs flushed and environment closed. '
              'Check acceptance reports for model quality.', flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    main()
    # close sim app
    simulation_app.close()
