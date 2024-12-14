import subprocess
from concurrent.futures import ThreadPoolExecutor
import itertools

TRAINING_RUN = 0

num_agents = [
    1, 
    3, 
    5, 
    10, 
    20
]

runs = [
    '', # plain.
    '-o True', # w/ obfuscation.
    '-t 3', # delay 3 timesteps
    '-t 7', # delay 7 timesteps.
    '-t 14', # delay 14 timesteps
    '-o True -t 3', # w/ obfuscation 3 timesteps.
    '-o True -t 14', # w/ obfuscation 14 timesteps
]

def train(num_agents, run):
    global TRAINING_RUN 
    TRAINING_RUN += 1
    command = f"python train_cooperative.py -f {num_agents} {run} -d cuda -a DQN"
    print(f"Running {TRAINING_RUN}: {command}")
    subprocess.run(command, shell=True)

def main():
    combos = itertools.product(num_agents, runs)

    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(lambda combo: train(*combo), combos)

    [print('DONE DONE DONE!!!') for i in range(20)]

if __name__ == "__main__":
    main()
