"""Train PPO in the GRPO_CUDA-style environment on the true FA-AFDM rate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from fa_afdm_dreamerv3.channel import FAAFDMConfig  # noqa: E402
from fa_afdm_dreamerv3.miso_style_afdm_env import MISOStyleAFDMConfig, MISOStyleFAAFDMEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO on GRPO_CUDA-style AFDM navigation or single-step position optimisation."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--resume-run", type=Path, default=None,
                        help="Continue a completed same-configuration run; timesteps is cumulative.")
    parser.add_argument("--resume-learning-rate-override", action="store_true",
                        help="Explicit LR-only diagnostic branch from a completed run; preserve Adam moments.")
    parser.add_argument("--resume-rollout-steps-override", action="store_true",
                        help="Explicit rollout-length diagnostic branch; environment horizon is unchanged.")
    parser.add_argument("--resume-gae-lambda-override", action="store_true",
                        help="Explicit GAE-lambda-only ablation branch from a completed run.")
    parser.add_argument("--checkpoint-freq", type=int, default=0,
                        help="Save model and observation/reward statistics after PPO updates at this interval.")
    parser.add_argument("--resume-terminal-reward-override", action="store_true",
                        help="Explicit terminal-potential ablation; recalibrate reward RMS in every branch.")
    parser.add_argument("--resume-fixed-start-override", action="store_true",
                        help="Explicit reset-distribution branch; switch to the geometric centre without changing PPO or reward.")
    parser.add_argument("--critic-adaptation-steps", type=int, default=0,
                        help="Fresh on-policy interactions with frozen actor and normal critic updates before PPO resumes.")
    parser.add_argument("--terminal-potential-alpha", type=float, default=1.0 / 1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--recurrent", action="store_true",
                        help="Use an LSTM PPO policy; observations stay channel plus position.")
    parser.add_argument("--bounded-beta-policy", action="store_true",
                        help="Use an exact bounded Beta actor so sampled and executed actions share one density.")
    parser.add_argument("--mean-precision-beta", action="store_true",
                        help="Parameterize the bounded Beta with separate mean and log-precision heads.")
    parser.add_argument("--mixture-beta-components", type=int, default=0,
                        help="Use this many exact bounded Beta components in one PPO policy (0 disables).")
    parser.add_argument("--beta-initial-concentration", type=float, default=None,
                        help="Optional symmetric Beta alpha=beta initial value for bounded-policy exploration scale.")
    parser.add_argument("--strict-action-bounds", action="store_true",
                        help="Fail on any out-of-box policy action instead of silently clipping it.")
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--policy-curve-freq", type=int, default=0,
                        help="Record deterministic current-policy terminal rate every N training steps (0 disables).")
    parser.add_argument("--episode-steps", type=int, default=50)
    parser.add_argument("--direct-position-action", action="store_true",
                        help="Map one bounded action directly to the four physical coordinates; use with --episode-steps 1.")
    parser.add_argument("--continuing-task", action="store_true",
                        help="Keep the last executed position as the next state; never reset after a step.")
    parser.add_argument("--evaluation-steps", type=int, default=100,
                        help="Finite rollout length used only to inspect a continuing frozen policy.")
    parser.add_argument("--include-time-feature", action="store_true",
                        help="Append t/T to observations for finite-horizon position-control episodes.")
    parser.add_argument("--finite-horizon-terminal", action="store_true",
                        help="Treat the task horizon as terminal, without value bootstrap.")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=2,
                        help="CPU threads for the small PPO network; avoids oversubscription.")
    parser.add_argument("--rollout-steps", type=int, default=125,
                        help="Steps per parallel environment before one PPO update.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--gae-lambda", type=float, default=0.99)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--ent-coef-final", type=float, default=None,
                        help="Optional linear entropy-coefficient target for PPO exploration annealing.")
    parser.add_argument("--ent-coef-anneal-steps", type=int, default=0,
                        help="Entropy annealing horizon; zero means --timesteps.")
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="PPO value-loss coefficient (SB3 default: 0.5).")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="PPO gradient clipping norm (SB3 default: 0.5).")
    parser.add_argument("--use-sde", action="store_true",
                        help="Use PPO's state-dependent exploration noise instead of independent action noise.")
    parser.add_argument("--sde-sample-freq", type=int, default=-1,
                        help="Resample gSDE noise every this many vector steps; -1 holds it for one rollout.")
    parser.add_argument("--net-width", type=int, default=256,
                        help="Width of each of the two PPO actor/critic hidden layers.")
    parser.add_argument("--position-fourier-max-frequency", type=int, default=0,
                        help="If positive, encode the same four position inputs with sin/cos bases through this frequency.")
    parser.add_argument("--channel-phase-features", action="store_true",
                        help="Encode phase differences using existing CSI and position; no reward labels.")
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--log-std-final", type=float, default=None,
                        help="Optional linear exploration-std annealing target.")
    parser.add_argument("--log-std-anneal-steps", type=int, default=0,
                        help="Annealing horizon; zero means --timesteps.")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--normalize-reward-only", action="store_true",
                        help="Normalise returns for PPO's critic while preserving raw channel/position observations.")
    parser.add_argument("--n-subcarriers", type=int, default=128)
    parser.add_argument("--n-paths", type=int, default=20)
    parser.add_argument("--channel-memory", type=int, default=5)
    parser.add_argument("--carrier-wavelength", type=float, default=0.0107068735)
    parser.add_argument("--bandwidth", type=float, default=10e6)
    parser.add_argument("--transmit-power", type=float, default=1.5)
    parser.add_argument("--noise-power-dbm", type=float, default=-95.0)
    parser.add_argument("--tx-region", type=float, default=0.5)
    parser.add_argument("--rx-region", type=float, default=0.5)
    parser.add_argument("--action-step", type=float, default=0.025)
    parser.add_argument("--boundary-penalty", type=float, default=10.0)
    parser.add_argument("--terminal-rate-weight", type=float, default=0.0,
                        help="Extra weight of the true AFDM rate at step T only.")
    parser.add_argument("--reward-mode", choices=["rate", "rate_delta", "rate_potential", "terminal_potential"], default="rate",
                        help="rate: dense current rate; rate_delta: exact final-rate potential shaping.")
    parser.add_argument("--rate-potential-scale", type=float, default=0.2)
    parser.add_argument("--dense-rate-coefficient", type=float, default=0.0,
                        help="Small auxiliary coefficient for (true rate - dense-rate-baseline) at every step.")
    parser.add_argument("--dense-rate-baseline", type=float, default=15.0)
    parser.add_argument("--rate-target", type=float, default=None,
                        help="Optional rate threshold used only by smooth monotone utility shaping.")
    parser.add_argument("--target-bonus", type=float, default=0.0,
                        help="Strength of the smooth utility preference near --rate-target.")
    parser.add_argument("--target-temperature", type=float, default=2.0)
    parser.add_argument("--fixed-center-start", action="store_true",
                        help="Start every episode at the feasible-region centre; this is not an optimum hint.")
    parser.add_argument("--train-start-pool", type=Path, default=None,
                        help="JSON [count,4] random physical-position pool used only by training environments.")
    parser.add_argument("--train-start-pool-probability", type=float, default=1.0,
                        help="Probability of a pool reset; remaining training resets are uniform random positions.")
    parser.add_argument("--channel-seed", type=int, default=7)
    parser.add_argument("--channel-gain-scale", type=float, default=1.5915494309189537e-5)
    parser.add_argument("--link-distance-min-m", type=float, default=500.0)
    parser.add_argument("--link-distance-max-m", type=float, default=1200.0)
    parser.add_argument("--path-loss-ref-distance-m", type=float, default=100.0)
    parser.add_argument("--path-loss-exponent", type=float, default=2.0)
    parser.add_argument("--fixed-doppler-hz", type=float, default=1297.193703548369)
    parser.add_argument("--gpu-rate-batch-size", type=int, default=0)
    parser.add_argument("--gpu-rate-device", default="cuda")
    parser.add_argument("--gpu-rate-backend", choices=["local", "rpc"], default="local")
    parser.add_argument("--gpu-rate-socket", default="")
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed: int, monitor_dir: Path, *, batched: bool,
             reset_position_pool: tuple[tuple[float, float, float, float], ...] | None = None,
             reset_position_pool_probability: float = 1.0):
    def factory():
        channel = FAAFDMConfig(
            n_subcarriers=args.n_subcarriers, n_paths=args.n_paths, channel_memory=args.channel_memory,
            carrier_wavelength=args.carrier_wavelength, bandwidth=args.bandwidth,
            transmit_power=args.transmit_power, tx_region=args.tx_region, rx_region=args.rx_region,
            noise_power_dbm=args.noise_power_dbm, channel_gain_scale=args.channel_gain_scale,
            link_distance_min_m=args.link_distance_min_m, link_distance_max_m=args.link_distance_max_m,
            path_loss_ref_distance_m=args.path_loss_ref_distance_m,
            path_loss_exponent=args.path_loss_exponent, fixed_doppler_hz=args.fixed_doppler_hz,
            seed=args.channel_seed,
        )
        env = MISOStyleFAAFDMEnv(
            channel,
            MISOStyleAFDMConfig(
                max_steps=args.episode_steps, action_step=args.action_step,
                direct_position_action=args.direct_position_action,
                continuing_task=args.continuing_task,
                include_time_feature=args.include_time_feature,
                finite_horizon_terminal=args.finite_horizon_terminal,
                boundary_penalty=args.boundary_penalty, terminal_rate_weight=args.terminal_rate_weight,
                reward_mode=args.reward_mode,
                rate_potential_scale=getattr(args,'rate_potential_scale',0.2),
                terminal_potential_alpha=getattr(args, 'terminal_potential_alpha', 1.0 / 1.2),
                dense_rate_coefficient=args.dense_rate_coefficient,
                dense_rate_baseline=args.dense_rate_baseline,
                rate_target=args.rate_target, target_bonus=args.target_bonus,
                target_temperature=args.target_temperature,
                strict_action_bounds=args.strict_action_bounds,
                random_reset=not args.fixed_center_start, reset_position_pool=reset_position_pool,
                reset_position_pool_probability=reset_position_pool_probability, seed=seed,
                gpu_rate_batch_size=args.gpu_rate_batch_size if batched else 0,
                gpu_rate_device=args.gpu_rate_device, gpu_rate_backend=args.gpu_rate_backend,
                gpu_rate_socket=args.gpu_rate_socket,
            ),
        )
        return Monitor(
            env,
            filename=str(monitor_dir / f"monitor_{seed}.csv"),
            info_keywords=("rate", "best_rate"),
        )

    return factory


def main() -> None:
    args = parse_args()
    import torch
    torch.set_num_threads(args.torch_threads)
    if args.reward_mode == 'terminal_potential':
        if args.gamma != 1 or not args.finite_horizon_terminal or args.continuing_task or not args.include_time_feature:
            raise ValueError('Terminal potential requires gamma=1, observed time and finite termination')
        if not 0 <= args.terminal_potential_alpha <= 1 or args.terminal_rate_weight <= 0:
            raise ValueError('Terminal potential requires alpha in [0,1] and positive terminal weight')
        if args.dense_rate_coefficient != 0 or args.target_bonus != 0:
            raise ValueError('Terminal-potential ablation cannot add another reward objective')
    if args.resume_terminal_reward_override and (not args.resume_run or args.reward_mode != 'terminal_potential'):
        raise ValueError('Reward override requires a parent and terminal_potential mode')
    if args.resume_fixed_start_override and (not args.resume_run or not args.fixed_center_start or args.train_start_pool is not None):
        raise ValueError('Fixed-start override requires resume, fixed centre and no start pool')
    if args.critic_adaptation_steps < 0 or args.critic_adaptation_steps % (args.rollout_steps * args.n_envs):
        raise ValueError('Critic adaptation must contain a nonnegative whole number of rollouts')
    if args.critic_adaptation_steps and not args.resume_terminal_reward_override:
        raise ValueError('Critic adaptation requires explicit reward ablation')
    if args.reward_mode == 'rate_potential':
        if args.gamma != 1 or not args.finite_horizon_terminal or args.continuing_task or not args.include_time_feature:
            raise ValueError('rate_potential requires gamma1, finite termination and observed time')
        if args.terminal_rate_weight != 0 or args.dense_rate_coefficient != 0 or args.target_bonus != 0:
            raise ValueError('rate_potential pilot must preserve the mean-rate objective without extra bonuses')
        if not np.isfinite(args.rate_potential_scale) or args.rate_potential_scale < 0:
            raise ValueError('Potential scale must be finite and nonnegative')
    if args.mean_precision_beta and not args.bounded_beta_policy:
        raise ValueError('--mean-precision-beta requires --bounded-beta-policy')
    resume_steps = 0
    if args.resume_learning_rate_override and not args.resume_run:
        raise ValueError('Learning-rate override requires --resume-run')
    if args.resume_rollout_steps_override and not args.resume_run:
        raise ValueError('Rollout override requires --resume-run')
    if args.resume_gae_lambda_override and not args.resume_run:
        raise ValueError('GAE-lambda override requires --resume-run')
    if args.resume_run:
        previous = json.loads((args.resume_run / 'config.json').read_text())
        if bool(previous.get('mean_precision_beta',False)) != args.mean_precision_beta:
            raise ValueError('Resume cannot change the policy parameterization')
        permitted = {'run_dir','timesteps','resume_run','checkpoint_freq','eval_freq','eval_episodes',
                     'gpu_rate_socket','device','resume_learning_rate_override','resume_rollout_steps_override',
                     'resume_terminal_reward_override', 'critic_adaptation_steps', 'resume_fixed_start_override'}
        if args.resume_fixed_start_override:
            permitted.add('fixed_center_start')
        if args.resume_terminal_reward_override:
            permitted.update({'reward_mode', 'terminal_rate_weight', 'terminal_potential_alpha'})
        if args.resume_rollout_steps_override:
            if args.rollout_steps <= 0:
                raise ValueError('Rollout steps must be positive')
            permitted.add('rollout_steps')
        if args.resume_learning_rate_override:
            if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
                raise ValueError('Learning rate must be finite and positive')
            permitted.add('learning_rate')
        if args.resume_gae_lambda_override:
            if not np.isfinite(args.gae_lambda) or not (0.0 <= args.gae_lambda <= 1.0):
                raise ValueError('GAE lambda must be in [0, 1]')
            permitted.add('gae_lambda')
        def resume_value_matches(key: str, previous_value) -> bool:
            current_value = getattr(args, key, previous_value)
            # A start pool is an input dataset.  Permit equivalent relative and
            # absolute spellings of the same file, but never a different pool.
            if key == "train_start_pool" and previous_value is not None and current_value is not None:
                return Path(previous_value).resolve() == Path(current_value).resolve()
            return current_value == previous_value

        differences = [key for key,value in previous.items()
                       if key not in permitted and not resume_value_matches(key, value)]
        if differences:
            raise ValueError(f'Resume requires unchanged training settings: {differences}')
        if not (args.resume_run / 'final_model.zip').exists():
            raise ValueError('Resume requires a completed training checkpoint.')
        resume_steps = int(previous['timesteps'])
        if args.timesteps <= resume_steps:
            raise ValueError('Cumulative timesteps must exceed the previous run.')
        if (args.timesteps-resume_steps) % (args.rollout_steps*args.n_envs):
            raise ValueError('Resume budget must contain an exact number of complete on-policy rollouts')
    if args.timesteps <= 0 or args.n_envs <= 0 or args.episode_steps <= 0:
        raise ValueError("--timesteps, --n-envs, and --episode-steps must be positive.")
    if args.direct_position_action and args.episode_steps != 1:
        raise ValueError("--direct-position-action is a T=1 experiment; set --episode-steps 1.")
    if args.continuing_task and args.direct_position_action:
        raise ValueError("A continuing task uses incremental actions, not --direct-position-action.")
    if args.gpu_rate_batch_size and args.gpu_rate_batch_size != args.n_envs:
        raise ValueError("--gpu-rate-batch-size must equal --n-envs for synchronized GPU batches.")

    args.run_dir.mkdir(parents=True, exist_ok=False)
    monitor_dir = args.run_dir / "monitor"
    monitor_dir.mkdir()
    eval_monitor_dir = args.run_dir / "eval_monitor"
    eval_monitor_dir.mkdir()
    (args.run_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8"
    )
    if args.resume_run:
        for filename in ['training_interaction_rates.csv','episode_average_rates.csv',
                         'episode_terminal_rates.csv','policy_training_curve.csv','ppo_updates.jsonl']:
            shutil.copyfile(args.resume_run / filename, args.run_dir / filename)
    set_random_seed(args.seed)

    class ThreadedDummyVecEnv(DummyVecEnv):
        """Step peer environments concurrently so AFDM rates form one CUDA batch."""

        def __init__(self, env_fns):
            super().__init__(env_fns)
            self._workers = ThreadPoolExecutor(max_workers=self.num_envs)

        def step_wait(self):
            results = list(self._workers.map(
                lambda index: self.envs[index].step(self.actions[index]), range(self.num_envs)
            ))
            for index, (obs, reward, terminated, truncated, info) in enumerate(results):
                self.buf_rews[index] = reward
                self.buf_infos[index] = info
                self.buf_dones[index] = terminated or truncated
                self.buf_infos[index]["TimeLimit.truncated"] = truncated and not terminated
                if self.buf_dones[index]:
                    self.buf_infos[index]["terminal_observation"] = obs
                    obs, self.reset_infos[index] = self.envs[index].reset()
                self._save_obs(index, obs)
            return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

        def close(self):
            self._workers.shutdown(wait=True)
            return super().close()

    reset_position_pool = None
    if args.train_start_pool is not None:
        raw_pool = json.loads(args.train_start_pool.read_text(encoding="utf-8"))
        pool = np.asarray(raw_pool, dtype=np.float64)
        if pool.ndim != 2 or pool.shape[1] != 4 or len(pool) == 0:
            raise ValueError("--train-start-pool must contain a nonempty JSON [count,4] array")
        if not np.isfinite(pool).all() or np.max(np.abs(pool)) > 0.2500001:
            raise ValueError("--train-start-pool positions must lie within [-0.25, 0.25] m")
        reset_position_pool = tuple(tuple(float(value) for value in row) for row in pool)
        if not (0.0 <= args.train_start_pool_probability <= 1.0):
            raise ValueError("--train-start-pool-probability must be in [0, 1]")
    env_fns = [
        make_env(args, args.seed + index, monitor_dir, batched=args.gpu_rate_batch_size > 1,
                 reset_position_pool=reset_position_pool,
                 reset_position_pool_probability=args.train_start_pool_probability)
        for index in range(args.n_envs)
    ]
    env = ThreadedDummyVecEnv(env_fns) if args.gpu_rate_batch_size > 1 else DummyVecEnv(env_fns)
    eval_env = DummyVecEnv([make_env(args, args.seed + 100_000, eval_monitor_dir, batched=False)])
    if args.no_normalize and args.normalize_reward_only:
        raise ValueError("--no-normalize and --normalize-reward-only are mutually exclusive.")
    if args.recurrent and args.bounded_beta_policy:
        raise ValueError("The available Beta policy is feed-forward; use either --recurrent or --bounded-beta-policy.")
    if args.mixture_beta_components and args.bounded_beta_policy:
        raise ValueError("Choose either --bounded-beta-policy or --mixture-beta-components.")
    if args.mixture_beta_components and args.recurrent:
        raise ValueError("The mixture Beta policy is feed-forward; do not combine it with --recurrent.")
    if args.mixture_beta_components == 1:
        raise ValueError("--mixture-beta-components must be zero or at least two.")
    if args.normalize_reward_only:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        eval_env = VecNormalize(eval_env, training=False, norm_obs=False, norm_reward=False, clip_obs=10.0)
    elif not args.no_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    if args.resume_run and isinstance(env,VecNormalize):
        env = VecNormalize.load(str(args.resume_run / 'vecnormalize.pkl'),env.venv)
        env.training=True
        if args.resume_terminal_reward_override:
            from stable_baselines3.common.running_mean_std import RunningMeanStd
            env.ret_rms = RunningMeanStd(shape=())
            env.returns.fill(0.0)

    policy_kwargs = dict(
        log_std_init=args.log_std_init,
        ortho_init=False,
        net_arch=dict(pi=[args.net_width, args.net_width], vf=[args.net_width, args.net_width]),
    )
    if args.position_fourier_max_frequency > 0:
        from fa_afdm_dreamerv3.ppo_position_features import FourierPositionFeatures
        policy_kwargs["features_extractor_class"] = FourierPositionFeatures
        policy_kwargs["features_extractor_kwargs"] = {
            "max_frequency": args.position_fourier_max_frequency,
            "include_time_feature": args.include_time_feature,
        }
    if args.channel_phase_features:
        if not (args.no_normalize or args.normalize_reward_only):
            raise ValueError('Channel phase features require raw observations.')
        if args.position_fourier_max_frequency:
            raise ValueError('Choose one feature encoder.')
        from fa_afdm_dreamerv3.ppo_position_features import ChannelPhaseFeatures
        policy_kwargs['features_extractor_class'] = ChannelPhaseFeatures
        policy_kwargs['features_extractor_kwargs'] = dict(
            n_paths=args.n_paths, wavelength=args.carrier_wavelength,
            tx_region=args.tx_region, rx_region=args.rx_region,
            include_time_feature=args.include_time_feature)
    tensorboard_log = str(args.run_dir / "tensorboard") if importlib.util.find_spec("tensorboard") else None
    if tensorboard_log is None:
        print("TensorBoard is not installed; continuing with Monitor CSV and evaluation NPZ logs.")
    from fa_afdm_dreamerv3.audited_ppo import AuditedPPO
    algorithm = AuditedPPO
    policy_name = "MlpPolicy"
    if args.recurrent:
        from sb3_contrib import RecurrentPPO
        algorithm = RecurrentPPO
        policy_name = "MlpLstmPolicy"
        policy_kwargs["lstm_hidden_size"] = args.lstm_hidden_size
    elif args.bounded_beta_policy:
        from fa_afdm_dreamerv3.truncated_gaussian_policy import BetaPolicy
        policy_name = BetaPolicy
        if args.mean_precision_beta:
            from fa_afdm_dreamerv3.truncated_gaussian_policy import MeanPrecisionBetaPolicy
            policy_name = MeanPrecisionBetaPolicy
        if args.beta_initial_concentration is not None:
            policy_kwargs["initial_concentration"] = args.beta_initial_concentration
    elif args.mixture_beta_components:
        from fa_afdm_dreamerv3.truncated_gaussian_policy import MixtureBetaPolicy
        policy_name = MixtureBetaPolicy
        policy_kwargs["mixture_components"] = args.mixture_beta_components
    model = algorithm(
        policy_name,
        env,
        n_steps=args.rollout_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        tensorboard_log=tensorboard_log,
        use_sde=args.use_sde,
        sde_sample_freq=args.sde_sample_freq,
        verbose=1,
    )
    if args.resume_run:
        load_overrides = {'n_steps': args.rollout_steps} if args.resume_rollout_steps_override else {}
        model = algorithm.load(args.resume_run / 'final_model.zip', env=env, device=args.device, **load_overrides)
        if model.num_timesteps != resume_steps:
            raise ValueError('Checkpoint count does not match previous training budget.')
        if args.resume_learning_rate_override:
            from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate
            model.learning_rate = args.learning_rate
            model.lr_schedule = get_schedule_fn(args.learning_rate)
            update_learning_rate(model.policy.optimizer, args.learning_rate)
            (args.run_dir / 'parameter_branch.json').write_text(json.dumps({
                'parent_run': str(args.resume_run), 'branch_at_step': resume_steps,
                'changed_parameter': 'learning_rate',
                'old_value': previous['learning_rate'], 'new_value': args.learning_rate,
                'note': 'Shared history is not independent data. Adam moments and policy are retained.'
            }, indent=2))
        if args.resume_rollout_steps_override:
            assert model.n_steps == args.rollout_steps
            assert model.rollout_buffer.buffer_size == args.rollout_steps
            (args.run_dir / 'rollout_branch.json').write_text(json.dumps({
                'parent_run': str(args.resume_run), 'branch_at_step': resume_steps,
                'changed_parameter': 'rollout_steps',
                'old_value': previous['rollout_steps'], 'new_value': args.rollout_steps,
                'episode_steps_unchanged': args.episode_steps,
                'note': 'Fresh on-policy rollout buffer, retained model and optimizer. Shared prefix is not new data.'
            }, indent=2))
        if args.resume_gae_lambda_override:
            # SB3 restores this scalar from the checkpoint, so merely parsing a
            # new CLI value does not make an ablation effective after load().
            model.gae_lambda = float(args.gae_lambda)
            # RolloutBuffer owns the actual GAE recursion, and is likewise
            # reconstructed from the parent checkpoint by SB3 load().
            model.rollout_buffer.gae_lambda = float(args.gae_lambda)
            assert model.gae_lambda == float(args.gae_lambda)
            assert model.rollout_buffer.gae_lambda == float(args.gae_lambda)
            (args.run_dir / 'gae_lambda_branch.json').write_text(json.dumps({
                'parent_run': str(args.resume_run), 'branch_at_step': resume_steps,
                'changed_parameter': 'gae_lambda',
                'old_value': previous['gae_lambda'], 'new_value': args.gae_lambda,
                'note': 'Only advantage-return mixing differs; environment, reward, checkpoint and optimizer state are retained.'
            }, indent=2))
    model.diagnostics_path = str(args.run_dir / 'ppo_updates.jsonl')
    model.critic_adaptation_until = resume_steps + args.critic_adaptation_steps if args.critic_adaptation_steps else 0
    if args.resume_fixed_start_override:
        (args.run_dir / 'fixed_start_branch.json').write_text(json.dumps({
            'parent_run': str(args.resume_run), 'branch_at_step': resume_steps,
            'changed_parameter': 'fixed_center_start',
            'fixed_position_m': [0,0,0,0], 'position_selection': 'geometric centre; no reward-based selection',
            'policy_optimizer_and_reward_normalizer_retained': True,
        }, indent=2))
    if args.resume_terminal_reward_override:
        (args.run_dir / 'reward_branch.json').write_text(json.dumps({
            'parent_run': str(args.resume_run), 'branch_at_step': resume_steps,
            'alpha': args.terminal_potential_alpha, 'terminal_scale': args.terminal_rate_weight,
            'episode_return': 'terminal_scale*(R_T-alpha*R_0)-sum(boundary_cost)',
            'critic_adaptation_steps_included_in_budget': args.critic_adaptation_steps,
            'reward_normalizer_reset': True,
            'actor_initialization': 'unchanged parent actor; never receives reference positions',
        }, indent=2))
    model.checkpoint_freq = args.checkpoint_freq
    model.checkpoint_dir = str(args.run_dir / 'checkpoints')
    callback = None if (args.recurrent or args.continuing_task) else EvalCallback(
        eval_env,
        best_model_save_path=str(args.run_dir / "best_model"),
        log_path=str(args.run_dir / "evaluations"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    class EpisodeCurveCallback(BaseCallback):
        """Write the actual sampled-policy terminal AFDM rate for every episode."""

        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.handle = None

        def _on_training_start(self) -> None:
            self.handle = self.path.open("a" if args.resume_run else "w", encoding="utf-8")
            if not args.resume_run:
                self.handle.write("steps,env_index,terminal_rate,episode_return\n")

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for index, info in enumerate(infos):
                if (not args.continuing_task) and int(info.get("step", 0)) == args.episode_steps:
                    episode = info.get("episode", {})
                    self.handle.write(
                        f"{int(self.num_timesteps)},{index},{float(info['rate']):.12g},"
                        f"{float(episode.get('r', float('nan'))):.12g}\n"
                    )
            self.handle.flush()
            return True

        def _on_training_end(self) -> None:
            if self.handle is not None:
                self.handle.close()

    class InteractionRateCallback(BaseCallback):
        """Persist the true rate of every action actually executed in training."""

        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.handle = None
            self.pending = 0

        def _on_training_start(self) -> None:
            self.handle = self.path.open("a" if args.resume_run else "w", encoding="utf-8")
            if not args.resume_run:
                self.handle.write("steps,env_index,rate,episode_terminal,u_x,u_y,q_x,q_y\n")

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            first_step = int(self.num_timesteps) - len(infos) + 1
            for index, info in enumerate(infos):
                self.handle.write(
                    f"{first_step + index},{index},{float(info['rate']):.12g},"
                    f"{int((not args.continuing_task) and info.get('step', 0) == args.episode_steps)},"
                    f"{float(info['u'][0]):.12g},{float(info['u'][1]):.12g},"
                    f"{float(info['q'][0]):.12g},{float(info['q'][1]):.12g}\n"
                )
                self.pending += 1
            if self.pending >= 1000:
                self.handle.flush()
                self.pending = 0
            return True

        def _on_training_end(self) -> None:
            if self.handle is not None:
                self.handle.close()

    class EpisodeAverageRateCallback(BaseCallback):
        """Write physical average and terminal AFDM rates for each episode."""

        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.handle = None
            self.rate_sums: np.ndarray | None = None

        def _on_training_start(self) -> None:
            self.handle = self.path.open("a" if args.resume_run else "w", encoding="utf-8")
            if not args.resume_run:
                self.handle.write("steps,env_index,average_rate,terminal_rate\n")
            self.rate_sums = np.zeros(args.n_envs, dtype=np.float64)

        def _on_step(self) -> bool:
            if args.continuing_task:
                return True
            assert self.rate_sums is not None
            for index, info in enumerate(self.locals.get("infos", [])):
                self.rate_sums[index] += float(info["rate"])
                if int(info.get("step", 0)) == args.episode_steps:
                    self.handle.write(
                        f"{int(self.num_timesteps)},{index},{self.rate_sums[index] / args.episode_steps:.12g},"
                        f"{float(info['rate']):.12g}\n"
                    )
                    self.rate_sums[index] = 0.0
            self.handle.flush()
            return True

    class ActionStdAnnealCallback(BaseCallback):
        """Schedule PPO's sampled-action variance; deterministic deployment is unchanged."""

        def _on_step(self) -> bool:
            if args.log_std_final is None or not hasattr(self.model.policy, "log_std"):
                return True
            horizon = args.log_std_anneal_steps or args.timesteps
            fraction = min(1.0, float(self.num_timesteps) / max(1, horizon))
            value = (1.0 - fraction) * args.log_std_init + fraction * args.log_std_final
            self.model.policy.log_std.data.fill_(float(value))
            return True

    class EntropyAnnealCallback(BaseCallback):
        """Standard PPO entropy-coefficient schedule; it never changes actions directly."""

        def _on_step(self) -> bool:
            if args.ent_coef_final is None:
                return True
            horizon = args.ent_coef_anneal_steps or args.timesteps
            fraction = min(1.0, float(self.num_timesteps) / max(1, horizon))
            self.model.ent_coef = (1.0 - fraction) * args.ent_coef + fraction * args.ent_coef_final
            return True

    class DeterministicPolicyCurveCallback(BaseCallback):
        """Evaluate the *current* actor, never a historical best, at fixed steps."""

        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.handle = None

        def _on_training_start(self) -> None:
            self.handle = self.path.open("a" if args.resume_run else "w", encoding="utf-8")
            if not args.resume_run:
                self.handle.write("steps,terminal_rate,episode_return\n")

        def _on_step(self) -> bool:
            if args.policy_curve_freq <= 0 or self.num_timesteps % args.policy_curve_freq:
                return True
            curve_env = MISOStyleFAAFDMEnv(
                FAAFDMConfig(
                    n_subcarriers=args.n_subcarriers, n_paths=args.n_paths, channel_memory=args.channel_memory,
                    carrier_wavelength=args.carrier_wavelength, bandwidth=args.bandwidth,
                    transmit_power=args.transmit_power, tx_region=args.tx_region, rx_region=args.rx_region,
                    noise_power_dbm=args.noise_power_dbm, channel_gain_scale=args.channel_gain_scale,
                    link_distance_min_m=args.link_distance_min_m, link_distance_max_m=args.link_distance_max_m,
                    path_loss_ref_distance_m=args.path_loss_ref_distance_m,
                    path_loss_exponent=args.path_loss_exponent, fixed_doppler_hz=args.fixed_doppler_hz,
                    seed=args.channel_seed,
                ),
                MISOStyleAFDMConfig(max_steps=args.episode_steps, action_step=args.action_step,
                                    direct_position_action=args.direct_position_action,
                                    continuing_task=args.continuing_task,
                                    include_time_feature=args.include_time_feature,
                                    finite_horizon_terminal=args.finite_horizon_terminal,
                                    boundary_penalty=args.boundary_penalty,
                                    terminal_rate_weight=args.terminal_rate_weight,
                                    reward_mode=args.reward_mode,
                                    rate_potential_scale=args.rate_potential_scale,
                                    dense_rate_coefficient=args.dense_rate_coefficient,
                                    dense_rate_baseline=args.dense_rate_baseline,
                                    rate_target=args.rate_target, target_bonus=args.target_bonus,
                                    target_temperature=args.target_temperature,
                                    strict_action_bounds=args.strict_action_bounds,
                                    random_reset=not args.fixed_center_start, seed=args.seed + 50_000),
            )
            observation, _ = curve_env.reset()
            recurrent_state, episode_start, total_reward, info = None, np.ones((1,), dtype=bool), 0.0, {}
            for _ in range(args.evaluation_steps if args.continuing_task else args.episode_steps):
                policy_observation = env.normalize_obs(observation) if isinstance(env, VecNormalize) else observation
                action, recurrent_state = self.model.predict(
                    policy_observation, state=recurrent_state, episode_start=episode_start, deterministic=True
                )
                observation, reward, _, truncated, info = curve_env.step(np.asarray(action).reshape(-1))
                total_reward += float(reward)
                episode_start = np.array([truncated], dtype=bool)
            self.handle.write(f"{self.num_timesteps},{float(info['rate']):.12g},{total_reward:.12g}\n")
            self.handle.flush()
            return True

        def _on_training_end(self) -> None:
            if self.handle is not None:
                self.handle.close()

    class EpisodeStartSourceCallback(BaseCallback):
        """Audit only the reset distribution of this run; it never affects PPO."""

        def __init__(self, path: Path):
            super().__init__()
            self.path = path
            self.handle = None

        def _on_training_start(self) -> None:
            self.handle = self.path.open("w", encoding="utf-8")
            self.handle.write("steps,env_index,reset_source,terminal_rate\n")

        def _on_step(self) -> bool:
            for index, info in enumerate(self.locals.get("infos", [])):
                if int(info.get("step", 0)) == args.episode_steps:
                    self.handle.write(f"{int(self.num_timesteps)},{index},{info.get('reset_source','unknown')},"
                                      f"{float(info['rate']):.12g}\n")
            self.handle.flush()
            return True

        def _on_training_end(self) -> None:
            if self.handle is not None:
                self.handle.close()

    callbacks = CallbackList(
        [
            *([] if callback is None else [callback]),
            EpisodeCurveCallback(args.run_dir / "episode_terminal_rates.csv"),
            InteractionRateCallback(args.run_dir / "training_interaction_rates.csv"),
            EpisodeAverageRateCallback(args.run_dir / "episode_average_rates.csv"),
            EpisodeStartSourceCallback(args.run_dir / "episode_start_sources.csv"),
            *([] if args.bounded_beta_policy else [ActionStdAnnealCallback()]),
            EntropyAnnealCallback(),
            DeterministicPolicyCurveCallback(args.run_dir / "policy_training_curve.csv"),
        ]
    )
    has_progress_bar = (
        importlib.util.find_spec("tqdm") is not None and importlib.util.find_spec("rich") is not None
    )
    model.learn(total_timesteps=args.timesteps-resume_steps, callback=callbacks,
                reset_num_timesteps=not bool(args.resume_run), progress_bar=has_progress_bar)
    model.save(args.run_dir / "final_model")
    if isinstance(env, VecNormalize):
        env.save(str(args.run_dir / "vecnormalize.pkl"))

    # Deterministic frozen-policy evaluation on the true, unnormalised reward.
    evaluation = MISOStyleFAAFDMEnv(
        FAAFDMConfig(
            n_subcarriers=args.n_subcarriers, n_paths=args.n_paths, channel_memory=args.channel_memory,
            carrier_wavelength=args.carrier_wavelength, bandwidth=args.bandwidth,
            transmit_power=args.transmit_power, tx_region=args.tx_region, rx_region=args.rx_region,
            noise_power_dbm=args.noise_power_dbm, channel_gain_scale=args.channel_gain_scale,
            link_distance_min_m=args.link_distance_min_m, link_distance_max_m=args.link_distance_max_m,
            path_loss_ref_distance_m=args.path_loss_ref_distance_m,
            path_loss_exponent=args.path_loss_exponent, fixed_doppler_hz=args.fixed_doppler_hz,
            seed=args.channel_seed,
        ),
        MISOStyleAFDMConfig(max_steps=args.episode_steps, action_step=args.action_step,
                            direct_position_action=args.direct_position_action,
                            continuing_task=args.continuing_task,
                            include_time_feature=args.include_time_feature,
                            finite_horizon_terminal=args.finite_horizon_terminal,
                            boundary_penalty=args.boundary_penalty,
                            terminal_rate_weight=args.terminal_rate_weight,
                            reward_mode=args.reward_mode,
                            rate_potential_scale=args.rate_potential_scale,
                            terminal_potential_alpha=args.terminal_potential_alpha,
                            dense_rate_coefficient=args.dense_rate_coefficient,
                            dense_rate_baseline=args.dense_rate_baseline,
                            rate_target=args.rate_target, target_bonus=args.target_bonus,
                            target_temperature=args.target_temperature,
                            strict_action_bounds=args.strict_action_bounds,
                            random_reset=not args.fixed_center_start, seed=args.seed + 200_000),
    )
    returns = []
    terminal_rates = []
    for episode in range(args.eval_episodes):
        observation, _ = evaluation.reset(seed=args.seed + 200_000 + episode)
        if isinstance(env, VecNormalize):
            observation = env.normalize_obs(observation[None, :])[0]
        done = False
        total_reward = 0.0
        info = {}
        recurrent_state = None
        episode_start = np.ones((1,), dtype=bool)
        evaluation_steps = 0
        evaluation_horizon = args.evaluation_steps if args.continuing_task else args.episode_steps
        while not done and evaluation_steps < evaluation_horizon:
            action, recurrent_state = model.predict(
                observation, state=recurrent_state, episode_start=episode_start, deterministic=True
            )
            next_observation, reward, terminated, truncated, info = evaluation.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            episode_start = np.array([done], dtype=bool)
            if isinstance(env, VecNormalize):
                next_observation = env.normalize_obs(next_observation[None, :])[0]
            observation = next_observation
            evaluation_steps += 1
        returns.append(total_reward)
        terminal_rates.append(float(info["rate"]))
    result = {
        "environment": (
            "continuing AFDM continuous-position-control environment" if args.continuing_task else
            "single-step AFDM position-optimisation environment" if args.direct_position_action else
            "GRPO_CUDA-style multi-step AFDM navigation environment"
        ),
        "objective": "true fixed-channel AFDM rate",
        "frozen_policy_return_mean": float(np.mean(returns)),
        "frozen_policy_return_std": float(np.std(returns)),
        "frozen_policy_terminal_sum_rate_mean": float(np.mean(terminal_rates)),
        "frozen_policy_terminal_sum_rate_std": float(np.std(terminal_rates)),
    }
    (args.run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
