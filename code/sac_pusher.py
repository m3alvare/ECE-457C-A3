import gymnasium as gym
import os
import argparse
from multiprocessing import Process
from stable_baselines3 import SAC

def train(hyperparams, run_id, base_log_dir, base_model_dir):
    MAXTIMESTEPS = 2000000
    TIMESTEPS = 10000

    log_dir = f"{base_log_dir}/run_{run_id}"
    model_dir = f"{base_model_dir}/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    pid = os.getpid()

    env = gym.make("Pusher-v4", render_mode=None)

    model = SAC(
        "MlpPolicy", 
        env, 
        tensorboard_log=log_dir,
        **hyperparams
    )

    print(f"Training run id:{run_id}, pid: {pid}")
    print(f"Hyperparameters: {hyperparams}")
    
    episodes = 0
    while episodes*TIMESTEPS <= MAXTIMESTEPS:
        episodes += 1
        model.learn(total_timesteps=TIMESTEPS, log_interval=5, reset_num_timesteps=False)
        model.save(f"{model_dir}/SAC_{TIMESTEPS*episodes}")
        print(f"Training run id:{run_id}, pid: {pid}, at {episodes*TIMESTEPS}")

    print("Done training SAC pusher")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on Pusher-v4 with different hyperparameters")
    parser.add_argument("mode", choices=["train"], help="Mode to run: train")
    args = parser.parse_args()

    # Default hyperparameters
    default_params = {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "buffer_size": 100000,
        "batch_size": 256,
        "tau": 0.005,
    }

    # Hyperparameter variations
    hyperparams = [
        default_params,  # Run with default values
        {**default_params, "learning_rate": 0.0001},
        {**default_params, "learning_rate": 0.001},
        {**default_params, "gamma": 0.95},
        {**default_params, "gamma": 0.999},
        {**default_params, "buffer_size": 50000},
        {**default_params, "buffer_size": 200000},
        {**default_params, "batch_size": 128},
        {**default_params, "batch_size": 512},
        {**default_params, "tau": 0.001},
        {**default_params, "tau": 0.01},
    ]

    if args.mode == "train":
        base_log_dir = "logs/pusher/SAC"
        base_model_dir = "pusher_models/SAC"
        os.makedirs(base_log_dir, exist_ok=True)
        os.makedirs(base_model_dir, exist_ok=True)

        processes = []
        for test_num, params in enumerate(hyperparams):
            p = Process(target=train, args=(params, test_num, base_log_dir, base_model_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()