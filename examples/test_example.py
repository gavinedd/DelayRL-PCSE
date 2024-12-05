# Import the necessary classes from crop-gym
from pcse_gym.envs.common_env import PCSEEnv
from pcse.fileinput import CABOFileReader, YAMLCropDataProvider
from pcse.util import WOFOST80SiteDataProvider

# Create and configure a PCSE-Gym environment
env = PCSEEnv(
    model_config='Wofost80_NWLP_FD.conf',
    agro_config='../PCSE-Gym/pcse_gym/envs/configs/agro/potato_cropcalendar.yaml',
    crop_parameters=YAMLCropDataProvider(force_reload=True),
    site_parameters=WOFOST80SiteDataProvider(WAV=10,  # Initial amount of water in total soil profile [cm]
                                             NAVAILI=10,  # Amount of N available in the pool at initialization of the system [kg/ha]
                                             PAVAILI=50,  # Amount of P available in the pool at initialization of the system [kg/ha]
                                             KAVAILI=100,  # Amount of K available in the pool at initialization of the system [kg/ha]
                                             ),
    soil_parameters=CABOFileReader('../PCSE-Gym/pcse_gym/envs/configs/soil/ec3.CAB'),
)

# Reset/initialize the environment to obtain an initial observation
observation = env.reset()
# Define an action that does nothing
action = {
    'irrigation': 0,
    'N': 0,
    'P': 0,
    'K': 0,
}

# Apply the action to the environment
observation, reward, done, truncated, info = env.step(action)

# Print the observation and reward
print("Observation:", observation)
print("Reward:", reward)
