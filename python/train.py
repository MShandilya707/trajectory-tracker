import os
import sys
import argparse
import tomli
sys.path.insert(0, os.path.dirname(__file__))

from env.trajectory_env import TrajectoryTrackingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

def load_config(config_path):
    with open(config_path, "rb") as f:
        return tomli.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.toml")
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--delay", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    # Load base config
    config_full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    config = load_config(config_full_path)

    # Override with CLI args if provided
    traj = args.traj or config["env"].get("traj_type", "circle")
    noise = args.noise if args.noise is not None else config["env"]["noise_std"]
    delay = args.delay if args.delay is not None else config["env"]["delay_steps"]
    timesteps = args.timesteps if args.timesteps is not None else config["training"].get("timesteps", 300000)
    out_dir = args.out or "results/"

    print(f"[armtracker] Starting training for trajectory: {traj}")
    print(f"[armtracker] Noise: {noise}, Delay steps: {delay}, Timesteps: {timesteps}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("models/", exist_ok=True)
    os.makedirs("models/best/", exist_ok=True)

    def make_env():
        env = TrajectoryTrackingEnv(
            traj_type=traj, 
            noise_std=noise, 
            delay_steps=delay,
            config=config["env"],
            reward_config=config["reward"]
        )
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=4)
    eval_env = make_env()

    eval_callback = EvalCallback(eval_env, best_model_save_path="models/best/",
                                 log_path=out_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="models/",
                                             name_prefix=f"sac_checkpoint_{traj}")

    policy_kwargs = dict(net_arch=config["training"]["net_arch"])
    
    model = SAC("MlpPolicy", vec_env, 
                learning_rate=config["training"]["learning_rate"], 
                buffer_size=config["training"]["buffer_size"],
                batch_size=config["training"]["batch_size"], 
                tau=config["training"]["tau"], 
                gamma=config["training"]["gamma"], 
                ent_coef=config["training"]["ent_coef"],
                policy_kwargs=policy_kwargs, 
                device=config["training"]["device"], 
                verbose=1)

    print(f"[armtracker] Training model...")
    model.learn(total_timesteps=timesteps, callback=[eval_callback, checkpoint_callback])

    final_model_path = os.path.join("models", f"sac_final_{traj}")
    model.save(final_model_path)
    
    # Default model path
    model.save("models/sac_final")
    
    print(f"[armtracker] Training complete. Model saved to {final_model_path}")

if __name__ == "__main__":
    main()
