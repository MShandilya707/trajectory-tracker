import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tomli
sys.path.insert(0, os.path.dirname(__file__))

from env.trajectory_env import TrajectoryTrackingEnv
from stable_baselines3 import SAC

def load_config(config_path):
    with open(config_path, "rb") as f:
        return tomli.load(f)

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.toml")
    parser.add_argument("--model", type=str, default="models/sac_final")
    parser.add_argument("--traj", type=str, default=None)
    parser.add_argument("--out", type=str, default="results/")
    parser.add_argument("--ep", type=int, default=3)
    args = parser.parse_args()

    # Load base config
    config_full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    config = load_config(config_full_path)

    traj = args.traj or config["env"].get("traj_type", "circle")
    print(f"[armtracker] Evaluating model: {args.model} on trajectory: {traj}")
    
    os.makedirs(args.out, exist_ok=True)
    
    env = TrajectoryTrackingEnv(
        traj_type=traj, 
        noise_std=0.0, 
        delay_steps=0,
        config=config["env"],
        reward_config=config["reward"]
    )
    try:
        model = SAC.load(args.model, device="cpu")
    except Exception as e:
        print(f"[armtracker] Failed to load model from {args.model}. Evaluating with random policy. Error: {e}")
        model = None

    all_rmse = []
    all_max_error = []
    all_mean_jerk = []
    
    ee_pos_history = []
    target_pos_history = []
    pos_error_history = []
    reward_history = []
    
    for ep in range(args.ep):
        obs, _ = env.reset()
        done = False
        
        ep_ee_pos = []
        ep_target_pos = []
        ep_error = []
        ep_reward = []
        
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            ee_pos = obs[14:17]
            target_pos = obs[17:20]
            
            ep_ee_pos.append(ee_pos)
            ep_target_pos.append(target_pos)
            ep_error.append(info["pos_error"])
            ep_reward.append(reward)
            
        ep_ee_pos = np.array(ep_ee_pos)
        ep_target_pos = np.array(ep_target_pos)
        ep_error = np.array(ep_error)
        ep_reward = np.array(ep_reward)
        
        rmse = float(np.sqrt(np.mean(ep_error**2)))
        max_err = float(np.max(ep_error))
        
        all_rmse.append(rmse)
        all_max_error.append(max_err)
        
        if ep == 0:
            ee_pos_history = ep_ee_pos
            target_pos_history = ep_target_pos
            pos_error_history = ep_error
            reward_history = ep_reward

        print(f"[armtracker] Episode {ep+1}/{args.ep} - Mean Error: {np.mean(ep_error):.4f}, Max Error: {max_err:.4f}, RMSE: {rmse:.4f}")
        
    jerk = np.diff(ee_pos_history, n=3, axis=0) / (env.dt ** 3)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    mean_jerk = float(np.mean(jerk_mag))
    
    x_pos = ee_pos_history[:, 0]
    fft_vals = np.fft.rfft(x_pos)
    power = np.abs(fft_vals) ** 2
    if np.sum(power) > 0:
        spectral_purity = float(np.sum(power[:5]) / np.sum(power))
    else:
        spectral_purity = 0.0

    print(f"\n[armtracker] --- Summary ---")
    print(f"[armtracker] Mean RMSE: {np.mean(all_rmse):.4f}")
    print(f"[armtracker] Max Error: {np.max(all_max_error):.4f}")
    print(f"[armtracker] Mean Jerk: {mean_jerk:.2f}")
    print(f"[armtracker] Spectral Purity (X-axis): {spectral_purity:.4f}")

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"pH-NODE SAC Trajectory Tracker — {args.traj}", fontsize=16)
    
    time_steps = np.arange(len(pos_error_history)) * env.dt
    
    # 1. Tracking error over time
    axs[0, 0].plot(time_steps, pos_error_history, color='red', label='Error')
    mean_err = np.mean(pos_error_history)
    axs[0, 0].axhline(mean_err, color='darkred', linestyle='--', label=f'Mean: {mean_err:.3f}')
    axs[0, 0].fill_between(time_steps, pos_error_history, alpha=0.3, color='red')
    axs[0, 0].set_title("Tracking Error over Time")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Error (m)")
    axs[0, 0].legend()
    
    # 2. XY trajectory overlay
    axs[0, 1].plot(target_pos_history[:, 0], target_pos_history[:, 1], 'g--', label='Target')
    axs[0, 1].plot(ee_pos_history[:, 0], ee_pos_history[:, 1], 'b-', label='Achieved')
    axs[0, 1].set_aspect('equal', 'box')
    axs[0, 1].set_title("XY Trajectory Overlay")
    axs[0, 1].set_xlabel("X (m)")
    axs[0, 1].set_ylabel("Y (m)")
    axs[0, 1].legend()
    
    # 3. Per-axis tracking
    axs[0, 2].plot(time_steps, target_pos_history[:, 0], 'r--', alpha=0.5)
    axs[0, 2].plot(time_steps, ee_pos_history[:, 0], 'r-', label='X')
    axs[0, 2].plot(time_steps, target_pos_history[:, 1], 'g--', alpha=0.5)
    axs[0, 2].plot(time_steps, ee_pos_history[:, 1], 'g-', label='Y')
    axs[0, 2].plot(time_steps, target_pos_history[:, 2], 'b--', alpha=0.5)
    axs[0, 2].plot(time_steps, ee_pos_history[:, 2], 'b-', label='Z')
    axs[0, 2].set_title("Per-axis X/Y/Z Tracking")
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Position (m)")
    axs[0, 2].legend()
    
    # 4. Reward per step
    axs[1, 0].plot(time_steps, reward_history, color='purple')
    axs[1, 0].set_title("Reward per Step")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Reward")
    
    # 5. Jerk magnitude
    jerk_time = time_steps[3:]
    axs[1, 1].plot(jerk_time, jerk_mag, color='orange', label='Jerk Mag')
    axs[1, 1].axhline(mean_jerk, color='darkorange', linestyle='--', label=f'Mean: {mean_jerk:.1f}')
    axs[1, 1].set_title("Jerk Magnitude over Time")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Jerk (m/s^3)")
    axs[1, 1].legend()
    
    # 6. Spectral power of X-axis
    freqs = np.fft.rfftfreq(len(x_pos), d=env.dt)
    axs[1, 2].semilogy(freqs, power, color='teal')
    dom_freq_idx = np.argmax(power[1:]) + 1
    axs[1, 2].axvline(freqs[dom_freq_idx], color='red', linestyle='--', label=f'Dom Freq: {freqs[dom_freq_idx]:.2f} Hz')
    axs[1, 2].set_title("Spectral Power of X-axis EE Pos")
    axs[1, 2].set_xlabel("Frequency (Hz)")
    axs[1, 2].set_ylabel("Power")
    axs[1, 2].legend()
    
    plt.tight_layout()
    out_file = os.path.join(args.out, f"results_{args.traj}.png")
    plt.savefig(out_file, dpi=150)
    print(f"[armtracker] Saved evaluation figure to {out_file}")
    plt.close()

if __name__ == "__main__":
    evaluate()
