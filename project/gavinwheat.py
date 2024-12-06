import datetime
import gymnasium as gym
import numpy as np
import warnings

from pcse_gym.envs import common_env
from pcse_gym.utils import defaults
from pcse_gym.utils import process_pcse_output as process_pcse
from pcse_gym.envs.sb3 import ZeroNitrogenEnvStorage, StableBaselinesWrapper
from pcse_gym.envs.rewards import Rewards

# Suppress specific matplotlib legend warning
warnings.filterwarnings('ignore', message='No artists with labels found to put in legend')

class GavinWheat(gym.Env):
    """
    Environment with two sub-environments:
        (1) environment for applying actions of RL agent
        (2) a baseline environment (e.g. with zero nitrogen policy) for computing relative reward
    Year and location of episode is randomly picked from years and locations through reset().
    """
    
    # crop features: ['DVS', 'TGROWTH', 'LAI', 'NUPTT', 'TRAN', 'TNSOIL', 'TRAIN', 'TRANRF', 'WSO']
    # weather features: ['IRRAD', 'TMIN', 'RAIN']

    def __init__(
        self,
        crop_features=defaults.get_wofost_default_crop_features(),
        action_features=defaults.get_default_action_features(),
        weather_features=defaults.get_default_weather_features(),
        seed=0,
        costs_nitrogen=None,
        timestep=7,
        years=None,
        locations=None,
        action_space=gym.spaces.Box(0, np.inf, shape=(1,)),
        action_multiplier=1.0,
        reward=None,
        timestep_delay=0,
        selected_crop_features=None,
        selected_weather_features=None,
        *args,
        **kwargs
    ):
        self.crop_features = crop_features
        self.action_features = action_features
        self.weather_features = weather_features
        self.costs_nitrogen = costs_nitrogen
        self.years = [years] if isinstance(years, int) else years
        self.locations = [locations] if isinstance(locations, tuple) else locations
        self.action_multiplier = action_multiplier
        self.action_space = action_space
        self._timestep = timestep
        self.reward_function = reward
        self.selected_crop_features = selected_crop_features
        self.selected_weather_features = selected_weather_features


        if self.reward_function != "GRO":
            self._env_baseline = self._initialize_sb_wrapper(seed, *args, **kwargs)
        self._env = self._initialize_sb_wrapper(seed, *args, **kwargs)

        self.observation_space = self._get_observation_space()
        self.zero_nitrogen_env_storage = ZeroNitrogenEnvStorage()
        self.rewards = Rewards(kwargs.get("reward_var"), self.timestep, costs_nitrogen)
        
        # add support for observation delays
        self.timestep_delay = timestep_delay
        self.observation_buffer = [np.zeros(self.observation_space.shape) for _ in range(self.timestep_delay)]
        
        print(f"Observation space:\ncrop features {self.crop_features}\nweather features: {self.weather_features}")

        super().reset(seed=seed)
    def _initialize_sb_wrapper(self, seed, *args, **kwargs):
        return StableBaselinesWrapper(
            crop_features=self.crop_features,
            action_features=self.action_features,
            weather_features=self.weather_features,
            costs_nitrogen=self.costs_nitrogen,
            timestep=self.timestep,
            years=self.years[0],
            location=self.locations[0],
            action_space=self.action_space,
            action_multiplier=self.action_multiplier,
            seed=seed,
            *args,
            **kwargs
        )

    def _get_observation_space(self):
        selected_crop_len = len(self.selected_crop_features) if self.selected_crop_features else len(self.crop_features)
        selected_weather_len = len(self.selected_weather_features) if self.selected_weather_features else len(self.weather_features)
        
        nvars = (
            selected_crop_len
            + len(self.action_features)
            + selected_weather_len * self.timestep
        )
        return gym.spaces.Box(0, np.inf, shape=(nvars,))

    def filter_observation(self, obs):
        filtered_obs = {}
    
        # Filter crop features
        if self.selected_crop_features:
            for feature in self.selected_crop_features:
                if feature in obs:
                    filtered_obs[feature] = obs[feature]
    
        # Filter weather features
        if self.selected_weather_features:
            for feature in self.selected_weather_features:
                if feature in obs:
                    filtered_obs[feature] = obs[feature]
    
        # If no specific features are selected, return the full observation
        if not filtered_obs:
            return obs
        
        return filtered_obs


    def step(self, action):
        """
        Computes customized reward and populates info
        """

        obs, _, terminated, truncated, info = self._env.step(action)


        # filter to only desired features
        obs = self.filter_observation(obs)
        self.observation_buffer.append(obs)
        self.observation_buffer.pop(0)
        delayed_obs = self.get_delayed_observation()


        output = self.sb3_env.model.get_output()
        if self.timestep_delay > 0:
            obs, reward, growth = self.process_output(action, output, delayed_obs)
        else:
            obs, reward, growth = self.process_output(action, output, obs)




        if "reward" not in info.keys():
            info["reward"] = {}
        info["reward"][self.date] = reward
        if "growth" not in info.keys():
            info["growth"] = {}
        info["growth"][self.date] = growth

        # return obs, reward, terminated, truncated, info
        if self.timestep_delay > 0:
            return delayed_obs, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, truncated, info

    def get_delayed_observation(self):
        if len(self.observation_buffer) > 0:
            return self.observation_buffer[-1]
        else:
            return None

    def process_output(self, action, output, obs):

        if isinstance(action, np.ndarray):
            action = action.item()
        amount = action * self.action_multiplier
        reward, growth = self.get_reward_and_growth(output, amount)
        return obs, reward, growth

    def get_reward_and_growth(self, output, amount):
        output_baseline = []
        if self.reward_function != "GRO":
            zero_nitrogen_results = self.zero_nitrogen_env_storage.get_episode_output(
                self.baseline_env
            )
            # convert zero_nitrogen_results to pcse_output
            var_name = process_pcse.get_name_storage_organ(zero_nitrogen_results.keys())
            for k, v in zero_nitrogen_results[var_name].items():
                if k <= output[-1]["day"]:
                    filtered_dict = {"day": k, var_name: v}
                    output_baseline.append(filtered_dict)

        reward, growth = self.get_reward_func(output, amount, output_baseline)
        return reward, growth

    def get_reward_func(self, output, amount, output_baseline=None):
        if self.reward_function == "DEF":
            return self.rewards.default_winterwheat_reward(
                output, output_baseline, amount
            )
        elif self.reward_function == "GRO":
            return self.rewards.growth_storage_organ(output, amount)
        else:
            return self.rewards.default_winterwheat_reward(
                output, output_baseline, amount
            )
    def overwrite_year(self, year):
        self.years = year
        if self.reward_function != "GRO":
            self.baseline_env.agro_management = self.sb3_env.agmt.replace_years(year)
        self.sb3_env.agro_management = self.sb3_env.agmt.replace_years(year)

    def set_location(self, location):
        if self.reward_function != "GRO":
            self.baseline_env.loc = location
            self.baseline_env.weather_data_provider = (
                common_env.get_weather_data_provider(location)
            )
        self.sb3_env.loc = location
        self.sb3_env.weather_data_provider = common_env.get_weather_data_provider(
            location
        )

    def overwrite_location(self, location):
        self.locations = location
        self.set_location(location)

    def reset(self, seed=None, options=()):
        if isinstance(self.years, list):
            year = self.np_random.choice(self.years)
            if self.reward_function != "GRO":
                self.baseline_env.agro_management = self.sb3_env.agmt.replace_years(
                    year
                )
            self.sb3_env.agro_management = self.sb3_env.agmt.replace_years(year)

        if isinstance(self.locations, list):
            location = self.locations[self.np_random.choice(len(self.locations), 1)[0]]
            self.set_location(location)
        if self.reward_function != "GRO":
            self.baseline_env.reset(seed=seed)
        obs = self.sb3_env.reset(seed=seed)


        # reset observation buffer
        self.observation_buffer = [np.zeros(self.observation_space.shape) for _ in range(self.timestep_delay)]


        # TODO: check whether info should/could be filled
        info = {}

        # return obs, info
        if self.timestep_delay > 0:
            return self.get_delayed_observation(), info
        else:
            return obs, info

    def render(self, mode="human"):
        pass

    @property
    def sb3_env(self):
        return self._env

    @property
    def baseline_env(self):
        return self._env_baseline

    @property
    def date(self) -> datetime.date:
        return self.sb3_env.model.day

    @property
    def loc(self) -> datetime.date:
        return self.sb3_env.loc

    @property
    def timestep(self):
        return self._timestep
