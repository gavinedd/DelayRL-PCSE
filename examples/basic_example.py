from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import lib_programname
import sys
import os
import torch.nn as nn

from pcse_gym.envs.winterwheat import WinterWheat
from pcse_gym.envs.sb3 import get_policy_kwargs, get_model_kwargs
from pcse_gym.utils.eval import EvalCallback, determine_and_log_optimum
import pcse_gym.utils.defaults as defaults

path_to_program = lib_programname.get_path_executed_script()
rootdir = path_to_program.parents[0]

if rootdir not in sys.path:
    print(f'insert {os.path.join(rootdir)}')
    sys.path.insert(0, os.path.join(rootdir))

if os.path.join(rootdir, "pcse_gym") not in sys.path:
    sys.path.insert(0, os.path.join(rootdir, "pcse_gym"))


def train(log_dir, n_steps,
          crop_features=defaults.get_wofost_default_crop_features(),
          weather_features=defaults.get_default_weather_features(),
          action_features=defaults.get_default_action_features(),
          train_years=defaults.get_default_train_years(),
          test_years=defaults.get_default_test_years(),
          train_locations=defaults.get_default_location(),
          test_locations=defaults.get_default_location(),
          action_space=defaults.get_default_action_space(),
          pcse_model=0, agent=PPO, reward=None,
          seed=0, tag="Exp", costs_nitrogen=10.0,
          **kwargs):
    """
    Train a PPO agent (Stable Baselines3).

    Parameters
    ----------
    log_dir: directory where the (tensorboard) data will be saved
    n_steps: int, number of timesteps the agent spends in the environment
    crop_features: crop features
    weather_features: weather features
    action_features: action features
    train_years: train years
    test_years: test years
    train_locations: train locations (latitude,longitude)
    test_locations: test locations (latitude,longitude)
    action_space: action space
    pcse_model: 0 for LINTUL else WOFOST
    agent: one of {PPO, RPPO, DQN}
    reward: one of {DEF, GRO}
    seed: random seed
    tag: tag for tensorboard and friends
    costs_nitrogen: float, penalty for fertilization application

    """

    pcse_model_name = "LINTUL" if not pcse_model else "WOFOST"

    print(f'Train model {pcse_model_name} with {agent} algorithm and seed {seed}. Logdir: {log_dir}')
    hyperparams = {'batch_size': 64, 'n_steps': 2048, 'learning_rate': 0.0003, 'ent_coef': 0.0, 'clip_range': 0.3,
                   'n_epochs': 10, 'gae_lambda': 0.95, 'max_grad_norm': 0.5, 'vf_coef': 0.5,
                   'policy_kwargs': get_policy_kwargs(n_crop_features=len(crop_features),
                                                      n_weather_features=len(weather_features),
                                                      n_action_features=len(action_features))}
    hyperparams['policy_kwargs']['net_arch'] = dict(pi=[256, 256], vf=[256, 256])
    hyperparams['policy_kwargs']['activation_fn'] = nn.Tanh
    hyperparams['policy_kwargs']['ortho_init'] = False


    env_pcse_train = WinterWheat(crop_features=crop_features, action_features=action_features,
                                 weather_features=weather_features,
                                 costs_nitrogen=costs_nitrogen, years=train_years, locations=train_locations,
                                 action_space=action_space, action_multiplier=1.0, seed=seed,
                                 reward=reward, **get_model_kwargs(pcse_model), **kwargs)

    env_pcse_train = Monitor(env_pcse_train)

    device = kwargs.get('device')

    print('Using CPU!')

    env_pcse_train = VecNormalize(DummyVecEnv([lambda: env_pcse_train]), norm_obs=True, norm_reward=True,
                                  clip_obs=10., clip_reward=50., gamma=1)
    model = PPO('MlpPolicy', env_pcse_train, gamma=1, seed=seed, verbose=0, **hyperparams,
                tensorboard_log=log_dir, device=device)

    env_pcse_eval = WinterWheat(crop_features=crop_features, action_features=action_features,
                                weather_features=weather_features,
                                costs_nitrogen=costs_nitrogen, years=test_years, locations=test_locations,
                                action_space=action_space, action_multiplier=1.0, reward=reward,
                                **get_model_kwargs(pcse_model), **kwargs, seed=seed)
    # env_pcse_eval = ActionLimiter(env_pcse_eval, action_limit=4)

    tb_log_name = f'{tag}-{pcse_model_name}-Ncosts-{costs_nitrogen}-run'

    model.learn(total_timesteps=n_steps,
                callback=EvalCallback(env_eval=env_pcse_eval, test_years=test_years,
                                      train_years=train_years, train_locations=train_locations,
                                      test_locations=test_locations, seed=seed, pcse_model=pcse_model,
                                      **kwargs),
                tb_log_name=tb_log_name)


if __name__ == '__main__':
    seed = 0
    nsteps = 400000
    costs_nitrogen = 10.0
    environment = 0
    agent = "PPO"
    reward = "DEF"
    device = "cpu"

    pcse_model_name = "LINTUL" if not environment else "WOFOST"
    log_dir = os.path.join(rootdir, 'tensorboard_logs', f'{pcse_model_name}_experiments')
    print(f'train for {nsteps} steps with costs_nitrogen={costs_nitrogen} (seed={seed})')

    all_years = [*range(1990, 2022)]
    train_years = [year for year in all_years if year % 2 == 1]
    test_years = [year for year in all_years if year % 2 == 0]

    train_locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
    test_locations = [(52, 5.5), (48, 0)]

    tag = f'Seed-{seed}'

    train(log_dir, train_years=train_years, test_years=test_years,
          train_locations=train_locations,
          test_locations=test_locations,
          n_steps=nsteps, seed=seed,
          tag=tag, costs_nitrogen=costs_nitrogen,
          crop_features=defaults.get_default_crop_features(pcse_env=environment),
          weather_features=defaults.get_default_weather_features(),
          action_features=defaults.get_default_action_features(),
          action_space=defaults.get_default_action_space(),
          pcse_model=environment, agent=agent,
          reward=reward, device=device)
