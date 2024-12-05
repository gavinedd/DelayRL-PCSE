import gym
import numpy as np
from pcse_gym.envs.common_env import PCSEEnv
from gym.spaces import Box
from pcse.fileinput import YAMLCropDataProvider

class FlattenedPCSEEnv(PCSEEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Flatten observation space
        sample_obs = self.reset()
        self.observation_space = self._get_flattened_obs_space(sample_obs)

        # Define a flattened action space (continuous irrigation, N, P, K)
        self.action_space = Box(
            low=np.array([0, 0, 0, 0]),  # Min values for irrigation, N, P, K
            high=np.array([100, 100, 100, 100]),  # Max values for irrigation, N, P, K
            dtype=np.float32
        )

    def _get_flattened_obs_space(self, sample_obs):
        # Get the dimensions of the flattened observation
        flattened_dim = self._flatten_observation(sample_obs).shape[0]
        return Box(low=-np.inf, high=np.inf, shape=(flattened_dim,), dtype=np.float32)

    def _flatten_observation(self, obs):
        # Convert observation dict to a flat vector
        crop_model_values = np.concatenate([np.array(v) for v in obs['crop_model'].values()])
        weather_values = np.concatenate([np.array(v) for v in obs['weather'].values()])
        return np.concatenate([crop_model_values, weather_values])

    def reset(self):
        obs = super().reset()
        return self._flatten_observation(obs)

    def step(self, action):
        # Convert flat action vector back to dictionary
        action_dict = {
            'irrigation': action[0],
            'N': action[1],
            'P': action[2],
            'K': action[3],
        }
        obs, reward, done, truncated, info = super().step(action_dict)
        return self._flatten_observation(obs), reward, done, truncated, info

# Example usage
env = FlattenedPCSEEnv(
    model_config='Wofost80_NWLP_FD.conf',
    agro_config='../PCSE-Gym/pcse_gym/envs/configs/agro/potato_cropcalendar.yaml',
    crop_parameters=YAMLCropDataProvider(force_reload=True),
    site_parameters=WOFOST80SiteDataProvider(WAV=10, NAVAILI=10, PAVAILI=50, KAVAILI=100),
    soil_parameters=CABOFileReader('../PCSE-Gym/pcse_gym/envs/configs/soil/ec3.CAB'),
)

# PPO-compatible flattened observation and action spaces
obs = env.reset()
print(f"Flattened observation space: {env.observation_space}")
print(f"Flattened action space: {env.action_space}")
print(f"Initial observation: {obs}")