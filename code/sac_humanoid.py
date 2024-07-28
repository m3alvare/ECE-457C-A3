import gymnasium as gym
import os
import argparse

from multiprocessing import Process
from stable_baselines3 import SAC

def train(hyperparams, run_id, base_log_dir, base_model_dir):
    TIMESTEPS = 50000

    log_dir = f"{base_log_dir}/run_{run_id}"
    model_dir = f"{base_model_dir}/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # create env
    env = gym.make("Humanoid-v4", render_mode=None)

    # init model with env and hyperparameters
    model = SAC(
        "MlpPolicy", 
        env, 
        # verbose=1, 
        tensorboard_log=log_dir, 
        buffer_size=100000,
        **hyperparams
    )

    print(f"Training run id:{run_id}")
    
    # training loop
    episodes = 0
    while True:
        episodes += 1
        model.learn(total_timesteps=TIMESTEPS, log_interval=5, reset_num_timesteps=False)
        model.save(f"{model_dir}/SAC_{TIMESTEPS*episodes}")

def test(model_path):
    print("tester func called")
    env = gym.make("Humanoid-v4", render_mode="human")
    print(model_path)
    model = SAC.load(model_path, env=env)
    
    obs = env.reset()[0]
    done = False
    extra_steps = 250 # to see the humanoid fall 
    while True:
        action, _states  = model.predict(obs)
        obs, reward, done, truncated, info  = env.step(action)
        if done:
            extra_steps -= 1
            if extra_steps == 0:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test SAC on Humanoid-v4")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    parser.add_argument("--model-path", help="Path to the model to load for testing")
    args = parser.parse_args()

    # suite of hyperparameter to test
    hyperparams = [
        {"learning_rate": 0.0001, "gamma":0.99},
        {"learning_rate": 0.0003, "gamma":0.99},
        {"learning_rate": 0.0005, "gamma":0.99},
        {"learning_rate": 0.0003, "gamma":0.999},
        {"learning_rate": 0.0003, "gamma":0.99},
        {"learning_rate": 0.0003, "gamma":0.96},
    ]

    if args.mode == "train":
        # create log and model dirs 
        base_log_dir = "logs/humanoid/SAC"
        base_model_dir = "humanoid_models"
        os.makedirs(base_log_dir, exist_ok=True)
        os.makedirs(base_model_dir, exist_ok=True)

        # train test suite in parallel with multi processing
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