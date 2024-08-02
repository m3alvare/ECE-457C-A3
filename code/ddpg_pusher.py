import gymnasium as gym
import os
import argparse
import numpy as np
from multiprocessing import Process
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

def train(hyperparams, run_id, base_log_dir, base_model_dir):
    MAXTIMESTEPS = 2000000
    TIMESTEPS = 25000

    log_dir = f"{base_log_dir}/run_{run_id}"
    model_dir = f"{base_model_dir}/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    pid = os.getpid()

    env = gym.make("Pusher-v4", render_mode=None)
    # Set up noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=hyperparams['noise_std'] * np.ones(n_actions))

    # Remove 'noise_std' from hyperparams as it's not a direct DDPG parameter
    hyperparams.pop("noise_std")
    
    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        tensorboard_log=log_dir,
        **hyperparams
    )

    print(f"Training run id:{run_id}, pid: {pid}")
    print(f"Hyperparameters: {hyperparams}")
    
    episodes = 0
    while episodes*TIMESTEPS <= MAXTIMESTEPS:
        episodes += 1
        model.learn(total_timesteps=TIMESTEPS, log_interval=5, reset_num_timesteps=False)
        model.save(f"{model_dir}/DDPG_{TIMESTEPS*episodes}")
        print(f"Training run id:{run_id}, pid: {pid}, at {episodes*TIMESTEPS}")

    print("Done training DDPG pusher")

def test(model_path):
    print("tester func called")
    env = gym.make("Pusher-v4", render_mode="human")
    print(model_path)
    model = DDPG.load(model_path, env=env)
    
    for i in range(10):
        obs = env.reset()[0]
        done = False
        timesteps = 0 # as the learning env stops after 100 timesteps
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            timesteps += 1
            if timesteps == 100:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test DDPG on Pusher-v4")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    parser.add_argument("--model-path", help="Path to the model to load for testing")
    args = parser.parse_args()

    default_params = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "buffer_size": 100000,
        "batch_size": 128,
        "tau": 0.005,
        "noise_std": 0.1,
    }

    hyperparams = [
        default_params,
        {**default_params, "learning_rate": 0.0003},
        {**default_params, "learning_rate": 0.003},
        {**default_params, "gamma": 0.95},
        {**default_params, "gamma": 0.999},
        {**default_params, "buffer_size": 50000},
        {**default_params, "buffer_size": 200000},
        {**default_params, "batch_size": 64},
        {**default_params, "batch_size": 256},
        {**default_params, "tau": 0.001},
        {**default_params, "tau": 0.01},
        # {**default_params, "noise_std": 0.2}, both had poor performance
        # {**default_params, "noise_std": 0.3}, ^^
    ]

    if args.mode == "train":
        base_log_dir = "logs/pusher/DDPG"
        base_model_dir = "pusher_models/DDPG"
        os.makedirs(base_log_dir, exist_ok=True)
        os.makedirs(base_model_dir, exist_ok=True)

        processes = []
        for test_num, params in enumerate(hyperparams):
            p = Process(target=train, args=(params, test_num, base_log_dir, base_model_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    elif args.mode == "test":
        if not args.model_path:
            print("Please provide a model path for testing.")
        else:
            test(args.model_path)