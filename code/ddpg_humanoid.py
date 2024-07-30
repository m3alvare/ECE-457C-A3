import gymnasium as gym
import os
import argparse
import numpy as np
from multiprocessing import Process
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

def train(hyperparams, run_id, base_log_dir, base_model_dir):
    TIMESTEPS = 1_000_000  # 1 million timesteps

    log_dir = f"{base_log_dir}/run_{run_id}"
    model_dir = f"{base_model_dir}/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create environment
    env = gym.make("Humanoid-v4", render_mode=None)
    # Set up noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=hyperparams['noise_std'] * np.ones(n_actions))

    # Remove 'noise_std' from hyperparams as it's not a direct DDPG parameter
    hyperparams.pop("noise_std")


    # Initialize model
    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise,
        # verbose=1, 
        tensorboard_log=log_dir, 
        **hyperparams
    )

    print(f"Training run id:{run_id}")

    
    # Training
    # model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)
    model.learn(total_timesteps=TIMESTEPS)
    model.save(f"{model_dir}/DDPG_final")

def test(model_path):
    env = gym.make("Humanoid-v4", render_mode="human")

    model = DDPG.load(model_path, env=env)
    
    for i in range(10):
        obs = env.reset()[0]
        done = False
        extra_steps = 100 # to see the humanoid fall 
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                extra_steps -= 1
                if extra_steps == 0:
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test DDPG on Humanoid-v4")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    parser.add_argument("--model-path", help="Path to the model to load for testing")
    args = parser.parse_args()

    # Hyperparameter configurations to test
    hyperparams = [ 
        {"learning_rate": 1e-3, "gamma": 0.98, "buffer_size": 100000, "noise_std": 0.1},
        {"learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 200000, "noise_std": 0.2},
        {"learning_rate": 5e-4, "gamma": 0.995, "buffer_size": 150000, "noise_std": 0.15},
        {"learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 200000, "noise_std": 0.1, "batch_size": 128},
        {"learning_rate": 1e-4, "gamma": 0.98, "buffer_size": 100000, "noise_std": 0.2, "batch_size": 64},
    ]

    if args.mode == "train":
        base_log_dir = "logs/humanoid/DDPG"
        base_model_dir = "humanoid_models/DDPG"
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