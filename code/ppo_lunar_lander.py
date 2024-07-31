import gymnasium as gym
import os
import argparse
from multiprocessing import Process
from stable_baselines3 import PPO

def train(hyperparams, run_id, base_log_dir, base_model_dir):
    MAXTIMESTEPS = 2000000
    TIMESTEPS = 50000

    log_dir = f"{base_log_dir}/run_{run_id}"
    model_dir = f"{base_model_dir}/run_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    pid = os.getpid()

    # create env
    env = gym.make("LunarLander-v2")

    # init model with env and hyperparameters
    model = PPO(
        "MlpPolicy", 
        env, 
        tensorboard_log=log_dir, 
        **hyperparams
    )

    print(f"Training run id:{run_id}, pid: {pid}")
    
    # training loop
    episodes = 0
    while episodes*TIMESTEPS <= MAXTIMESTEPS:
        episodes += 1
        model.learn(total_timesteps=TIMESTEPS, log_interval=5, reset_num_timesteps=False)
        model.save(f"{model_dir}/PPO_{TIMESTEPS*episodes}")
        print(f"Training run id:{run_id}, pid: {pid}, at {episodes*TIMESTEPS}")

    print("Done training PPO lunar landing")


def test(model_path):
    print("tester func called")
    env = gym.make("LunarLander-v2", render_mode="human")
    print(model_path)
    model = PPO.load(model_path, env=env)
    
    for i in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                print(f"Episode finished. Total reward: {total_reward}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test PPO on LunarLander-v2")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    parser.add_argument("--model-path", help="Path to the model to load for testing")
    args = parser.parse_args()

    # suite of hyperparameters to test
    hyperparams = [
        # ALL DEFAULT
        {"learning_rate": 0.0003},

        # learning rate default 0.0003
        {"learning_rate": 0.0001},
        {"learning_rate": 0.0005},

        # gamma discount factor default
        {"gamma": 0.999},
        {"gamma": 0.96},

        # batch size default 2048
        {"batch_size":512},
        {"batch_size":1024}, #worst one

        # gae_lambda default 0.95
        {"gae_lambda":0.85},
        {"gae_lambda":0.99},

        # clip_range default 0.2
        {"clip_range":0.1}, #BEST ONE
        {"clip_range":0.3},
    ]

    if args.mode == "train":
        # create log and model dirs 
        base_log_dir = "logs/lunar_lander/PPO"
        base_model_dir = "lunar_lander_models/PPO"
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