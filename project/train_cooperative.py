import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import logging
import os
import sys
import argparse
import lib_programname
from sb3_contrib import RecurrentPPO

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first

from project.cooperative.market import AgriculturalMarket
from project.cooperative.resources import SharedResources
from project.cooperative.knowledge import KnowledgePool
from pcse_gym.utils import defaults
from pcse_gym.envs.sb3 import get_model_kwargs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from project.cooperative.environment import CooperativeFarmEnv
from project.cooperative.market import AgriculturalMarket
from project.cooperative.resources import SharedResources
from project.cooperative.knowledge import KnowledgePool
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

class DictToBoxWrapper(gym.ObservationWrapper):
    """Convert Dict observation space to Box space"""
    def __init__(self, env):
        super().__init__(env)
        
        # Calculate total size of flattened observation
        total_size = sum(
            np.prod(space.shape)
            for space in env.observation_space.spaces.values()
        )
        
        # Define new Box observation space with proper bounds
        self.observation_space = spaces.Box(
            low=-1.0,  # Change from -inf to -1.0
            high=1.0,  # Change from inf to 1.0
            shape=(total_size,),
            dtype=np.float32
        )
        
        # Store reference to unwrapped env for attribute access
        self._unwrapped = env.unwrapped
    
    def observation(self, obs):
        # Flatten and concatenate all observations
        flat_obs = []
        for key in sorted(obs.keys()):
            value = obs[key].astype(np.float32)
            value = np.clip(value, -1.0, 1.0)
            # Handle NaN and inf values
            value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
            flat_obs.append(value.flatten())
        
        # Concatenate all observations
        flat_obs = np.concatenate(flat_obs)
        flat_obs = np.clip(flat_obs, -1.0, 1.0)
        return flat_obs
        
    @property
    def date(self):
        """Get current date from unwrapped environment"""
        return self._unwrapped.date
        
    @property
    def year(self):
        """Get current year from unwrapped environment"""
        return self._unwrapped.date.year
        
    @property
    def loc(self):
        """Get current location from unwrapped environment"""
        return self._unwrapped.loc

def create_cooperative_env(farm_id, crop_features, action_features, weather_features,
                         costs_nitrogen, years, locations, action_space, seed,
                         reward, market, shared_resources, knowledge_pool, pcse_model=0, **kwargs):
    """Create a cooperative farm environment"""
    
    env = CooperativeFarmEnv(
        farm_id=farm_id,
        market=market,
        shared_resources=shared_resources,
        knowledge_pool=knowledge_pool,
        crop_features=crop_features,
        action_features=action_features,
        weather_features=weather_features,
        costs_nitrogen=costs_nitrogen,
        years=years,
        locations=locations,
        action_space=action_space,
        action_multiplier=1.0,
        seed=seed,
        reward=reward,
        **get_model_kwargs(pcse_model),
        **kwargs
    )
    
    wrapped_env = DictToBoxWrapper(env)
    return wrapped_env

def make_env(farm_id, crop_features, action_features, weather_features,
             costs_nitrogen, years, locations, action_space, seed,
             reward, market, shared_resources, knowledge_pool, pcse_model=0, **kwargs):
    """Create a function that creates and returns a new environment instance"""
    def _init():
        env = create_cooperative_env(
            farm_id=farm_id,
            crop_features=crop_features,
            action_features=action_features,
            weather_features=weather_features,
            costs_nitrogen=costs_nitrogen,
            years=years,
            locations=locations,
            action_space=action_space,
            seed=seed,
            reward=reward,
            market=market,
            shared_resources=shared_resources,
            knowledge_pool=knowledge_pool,
            pcse_model=pcse_model,
            **kwargs
        )
        env = Monitor(env)
        return env
    return _init

def get_cooperative_policy_kwargs(n_crop_features, n_weather_features, n_action_features):
    """Get policy network configuration for cooperative environment"""
    total_features = n_crop_features + n_weather_features + n_action_features + 3  # +3 for cooperative features
    return {
        'net_arch': [64, 64],
        'activation_fn': nn.Tanh,
        'normalize_images': False,
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': {}
    }

class CustomEvalCallback(BaseCallback):
    """Custom callback for evaluating and logging metrics."""
    
    def __init__(self, eval_env, test_years, train_years, train_locations, test_locations,
                 seed, pcse_model, eval_freq=10000, n_eval_episodes=5, **kwargs):
        super().__init__(verbose=1)
        self.eval_env = eval_env  # Store eval_env as instance variable
        self.test_years = test_years
        self.train_years = train_years
        self.train_locations = train_locations
        self.test_locations = test_locations
        self.seed = seed
        self.pcse_model = pcse_model
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.kwargs = kwargs
    
    def _on_step(self):
        """
        This method will be called by the model after each step.
        """
        # Only evaluate every n steps
        if self.n_calls % self.eval_freq != 0:
            return True
            
        # Evaluate the agent
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,  # Use stored eval_env
            n_eval_episodes=self.n_eval_episodes,
            render=False,
            deterministic=True,
            return_episode_rewards=True,
        )
        
        mean_reward = np.mean(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        # Log the metrics
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_length", mean_length)
        
        # Store best mean reward
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            
        self.last_mean_reward = mean_reward
        
        # Log additional info
        self.logger.record("eval/best_mean_reward", self.best_mean_reward)
        
        return True

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
          n_farms=3, **kwargs):
    """
    Train multiple cooperative PPO agents.
    """
    pcse_model_name = "LINTUL" if not pcse_model else "WOFOST"
    logger.info(f'Train cooperative model {pcse_model_name} with {agent} algorithm and seed {seed}. Logdir: {log_dir}')

    # Get features based on model type
    if pcse_model == 0:  # LINTUL
        crop_features = defaults.get_default_crop_features(pcse_env=0)
    else:  # WOFOST
        crop_features = defaults.get_wofost_default_crop_features()
    
    weather_features = defaults.get_default_weather_features()
    action_features = defaults.get_default_action_features()
    
    logger.info(f"Using features: crop={len(crop_features)}, weather={len(weather_features)}, action={len(action_features)}")
    logger.info(f"Total observation size: {len(crop_features) + len(weather_features) + len(action_features) + 3}")  # +3 for cooperative features

    # Initialize cooperative components
    market = AgriculturalMarket(
        base_price=100.0,
        supply_sensitivity=0.1,
        supply_decay=0.05,
        volatility=0.1,
        min_price=10.0
    )
    
    shared_resources = SharedResources(
        base_cost=costs_nitrogen,
        sharing_discount=0.2,
        max_sharing_farms=n_farms,
        min_cost=costs_nitrogen * 0.1
    )
    
    knowledge_pool = KnowledgePool(
        learning_rate=0.1,
        memory_size=1000,
        similarity_threshold=0.2
    )

    policy_kwargs = get_cooperative_policy_kwargs(
        n_crop_features=len(crop_features),
        n_weather_features=len(weather_features),
        n_action_features=len(action_features)
    )

    if agent in ('PPO', 'RPPO'):
        hyperparams = {
            'batch_size': 64,
            'n_steps': 2048,
            'learning_rate': 3e-4,
            'ent_coef': 0.0,
            'clip_range': 0.2,
            'n_epochs': 10,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'vf_coef': 0.5,
            'policy_kwargs': policy_kwargs
        }
    if agent == 'DQN':
        hyperparams = {
            'learning_rate': 3e-4,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
            'policy_kwargs': policy_kwargs
        }

    # Create environment creation functions for each farm
    env_fns = []
    for farm_id in range(n_farms):
        env_fn = make_env(
            farm_id=farm_id,
            crop_features=crop_features,
            action_features=action_features,
            weather_features=weather_features,
            costs_nitrogen=costs_nitrogen,
            years=train_years,
            locations=train_locations,
            action_space=action_space,
            seed=seed + farm_id,
            reward=reward,
            market=market,
            shared_resources=shared_resources,
            knowledge_pool=knowledge_pool,
            pcse_model=pcse_model,
            **kwargs
        )
        env_fns.append(env_fn)

    device = kwargs.get('device')
    if device == 'cuda':
        print('CUDA not available... Using CPU!') if not torch.cuda.is_available() else print('using CUDA!')
    else:
        print('Using CPU!')
    
    # Create vectorized environment for training
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                          clip_obs=10., clip_reward=50., gamma=1)
    
    # Create single evaluation environment
    eval_env = create_cooperative_env(
        farm_id=0,  # Use first farm for evaluation
        crop_features=crop_features,
        action_features=action_features,
        weather_features=weather_features,
        costs_nitrogen=costs_nitrogen,
        years=test_years,
        locations=test_locations,
        action_space=action_space,
        seed=seed,
        reward=reward,
        market=AgriculturalMarket(),  # Fresh market for evaluation
        shared_resources=SharedResources(base_cost=costs_nitrogen),
        knowledge_pool=KnowledgePool(),
        pcse_model=pcse_model,
        **kwargs
    )
    
    # Create and train model
    if agent == 'PPO':
        model = PPO('MlpPolicy', vec_env, gamma=1, seed=seed, verbose=1, **hyperparams,
                    tensorboard_log=log_dir, device=device)
    elif agent == 'DQN':
        model = DQN('MlpPolicy', vec_env, gamma=1, seed=seed, verbose=1, **hyperparams,
                    tensorboard_log=log_dir, device=device)
    elif agent == 'RPPO':
        model = RecurrentPPO('MlpLstmPolicy', vec_env, gamma=1, seed=seed, verbose=1, **hyperparams,
                             tensorboard_log=log_dir, device=device)
    else:
        model = PPO('MlpPolicy', vec_env, gamma=1, seed=seed, verbose=1, **hyperparams,
                    tensorboard_log=log_dir, device=device)

    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        test_years=test_years,
        train_years=train_years,
        train_locations=train_locations,
        test_locations=test_locations,
        seed=seed,
        pcse_model=pcse_model,
        **kwargs
    )

    tb_log_name = f'{tag}-{pcse_model_name}-Cooperative-{n_farms}farms-Ncosts-{costs_nitrogen}-run'

    model.learn(total_timesteps=n_steps,
                callback=eval_callback,
                tb_log_name=tb_log_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="Set seed")
    parser.add_argument("-n", "--nsteps", type=int, default=400000, help="Number of steps")
    parser.add_argument("-c", "--costs_nitrogen", type=float, default=10.0, help="Costs for nitrogen")
    parser.add_argument("-e", "--environment", type=int, default=0,
                        help="Crop growth model. 0 for LINTUL-3, 1 for WOFOST")
    parser.add_argument("-a", "--agent", type=str, default="PPO", help="RL agent. PPO, RPPO, or DQN.")
    parser.add_argument("-r", "--reward", type=str, default="DEF", help="Reward function. DEF, or GRO")
    parser.add_argument("-f", "--farms", type=int, default=3, help="Number of cooperative farms")
    parser.add_argument('-d', "--device", type=str, default="cpu")

    parser.set_defaults(measure=True, vrr=False)

    args = parser.parse_args()
    pcse_model_name = "LINTUL" if not args.environment else "WOFOST"
    log_dir = os.path.join(rootdir, 'tensorboard_logs', f'{pcse_model_name}_cooperative_experiments')
    print(f'train for {args.nsteps} steps with costs_nitrogen={args.costs_nitrogen} (seed={args.seed})')

    all_years = [*range(1990, 2022)]
    train_years = [year for year in all_years if year % 2 == 1]
    test_years = [year for year in all_years if year % 2 == 0]

    train_locations = [(52, 5.5), (51.5, 5), (52.5, 6.0)]
    test_locations = [(52, 5.5), (48, 0)]

    tag = f'Seed-{args.seed}'

    train(log_dir, train_years=train_years, test_years=test_years,
          train_locations=train_locations,
          test_locations=test_locations,
          n_steps=args.nsteps, seed=args.seed,
          tag=tag, costs_nitrogen=args.costs_nitrogen,
          crop_features=defaults.get_default_crop_features(pcse_env=args.environment),
          weather_features=defaults.get_default_weather_features(),
          action_features=defaults.get_default_action_features(),
          action_space=defaults.get_default_action_space(),
          pcse_model=args.environment, agent=args.agent,
          reward=args.reward, device=args.device,
          n_farms=args.farms)