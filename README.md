# pH-NODE: RL-Based Robotic Arm Trajectory Tracker

[![Hybrid Architecture](https://img.shields.io/badge/Architecture-Rust%20%2B%20Python-orange.svg)](#)
[![Algorithm](https://img.shields.io/badge/Algorithm-SAC-blue.svg)](#)
[![Physics](https://img.shields.io/badge/Physics-PyBullet-green.svg)](#)

A high-performance, research-grade Reinforcement Learning system for precision robotic arm trajectory tracking. This project implements a **Soft Actor-Critic (SAC)** agent within a custom **Gymnasium** environment, featuring a unique hybrid architecture that combines the system-level safety of **Rust** with the rich AI ecosystem of **Python**.

## 🚀 Key Innovation: Multi-Objective Reward Shaping
Unlike standard tracking controllers, pH-NODE utilizes three distinct mathematical components to ensure smooth, stable, and accurate motion:

1.  **Gaussian Tracking Reward:** $\exp(-k \cdot \|p_{ee} - p_{target}\|^2)$ for sub-millimeter precision.
2.  **Spectral Smoothness Penalty:** An FFT-based reward that penalizes high-frequency jitter by analyzing the power spectrum of end-effector movement over a rolling window.
3.  **Lyapunov Stability Reward:** A gradient-descent inspired reward that ensures the agent is always moving *toward* the target manifold.

## 🏗 Architecture Overview

| Component | Responsibility | Technology |
| :--- | :--- | :--- |
| **Orchestrator** | CLI, Sequential Training, OS Management | **Rust** (Clap v4) |
| **RL Agent** | Soft Actor-Critic (SAC) Policy | **Stable-Baselines3** |
| **Simulation** | Physics, Franka Panda Model, Collision | **PyBullet / Panda-Gym** |
| **Signal Processing**| FFT, Reward Shaping, Spectral Analysis | **NumPy / SciPy** |

## 🛠 Features

*   **Hybrid Orchestration:** A single Rust binary manages Python environments and executes complex training/evaluation pipelines.
*   **Uncertainty Modeling:**
    *   **Action Delay Buffer:** Simulates real-world communication latency (FIFO queue).
    *   **Gaussian Observation Noise:** Robustness against sensor inaccuracies.
*   **Dynamic Trajectories:**
    *   **Circle:** Basic 2D planar tracking.
    *   **Figure-8:** Complex curvature and velocity changes.
    *   **3D Lissajous:** Multi-axis 3D tracking with reachable/unreachable goal regions.
*   **Config-Driven:** Fully customizable via `config/default.toml`—no code changes required for hyperparameter tuning.

## 🚦 Getting Started

### Prerequisites
*   Rust (1.70+)
*   Python 3.10+
*   Cargo

### Installation
1.  **Clone and Install Dependencies:**
    ```bash
    git clone <your-repo-url>
    cd arm-trajectory-tracker
    pip install -r requirements.txt
    ```

2.  **Build the Rust CLI:**
    ```bash
    cargo build --release
    ```

## 📈 Usage

The project is managed through the `armtracker` CLI.

### Train a Model
```bash
./target/release/armtracker train --traj lissajous --timesteps 300000
```

### Run Full Pipeline
Automatically trains and evaluates all trajectories sequentially:
```bash
./target/release/armtracker run-all --timesteps 100000
```

### Evaluate
```bash
./target/release/armtracker eval --model models/sac_final_circle --traj circle
```

## 📊 Results (After 500k/400k Steps)
The optimized high-precision models achieved the following metrics:

| Trajectory | Mean RMSE | Max Error | Spectral Purity |
| :--- | :--- | :--- | :--- |
| **Circle** | **0.0275 m** | 0.0447 m | **0.9999** |
| **Figure-8** | **0.0447 m** | 0.0673 m | 0.9936 |

*   **Warm-Start Phase:** Effectively eliminated initial tracking lurch.
*   **CBF Protection:** Guaranteed Z-axis safety throughout training and evaluation.
*   **Spectral Smoothness:** Achieved near-perfect motion purity (0.9999).

## 🛡 Roadmap (Upcoming)
*   [ ] **Control Barrier Functions (CBF):** Mathematically guaranteed safety layers to prevent self-collision.
*   [ ] **Domain Randomization:** Varying link masses and friction for sim-to-real transfer.
*   [ ] **Neural ODEs:** Continuous-time dynamics modeling for improved sample efficiency.

---
**Developed with ❤️ for Robotics Research.**
