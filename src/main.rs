use clap::{Parser, Subcommand};
use std::process::Command;
use std::process::exit;

#[derive(Parser)]
#[command(name = "armtracker")]
#[command(about = "RL Robotic Arm Trajectory Tracker", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the RL model
    Train {
        #[arg(long, default_value = "circle")]
        traj: String,
        
        #[arg(long, default_value_t = 0.005)]
        noise: f64,
        
        #[arg(long, default_value_t = 2)]
        delay: usize,
        
        #[arg(long, default_value_t = 300000)]
        timesteps: usize,
        
        #[arg(long, default_value = "results/")]
        out: String,
    },
    /// Evaluate the trained model
    Eval {
        #[arg(long, default_value = "models/sac_final")]
        model: String,
        
        #[arg(long, default_value = "circle")]
        traj: String,
        
        #[arg(long, default_value = "results/")]
        out: String,
        
        #[arg(long, default_value_t = 3)]
        episodes: usize,
    },
    /// Train and evaluate all 3 trajectories sequentially
    RunAll {
        #[arg(long, default_value_t = 200000)]
        timesteps: usize,
    },
}

fn run_python(script: &str, args: Vec<String>) {
    let python_cmd = if cfg!(windows) { "python" } else { "python3" };
    
    let mut command = Command::new(python_cmd);
    command.arg(format!("python/{}", script));
    
    for arg in args {
        command.arg(arg);
    }
    
    let status = command.status().expect("Failed to execute python command");
    if !status.success() {
        eprintln!("Python script {} failed with status: {}", script, status);
        exit(1);
    }
}

fn print_banner(title: &str) {
    let width = 60;
    let title_len = title.len();
    if title_len + 4 > width {
        println!("┌{}┐", "─".repeat(title_len + 2));
        println!("│ {} │", title);
        println!("└{}┘", "─".repeat(title_len + 2));
        return;
    }
    
    let padding = (width - title_len - 4) / 2;
    let pad_str = " ".repeat(padding);
    let pad_str_right = " ".repeat(width - title_len - 4 - padding);
    
    println!("┌{}┐", "─".repeat(width - 2));
    println!("│ {}{}{} │", pad_str, title, pad_str_right);
    println!("└{}┘", "─".repeat(width - 2));
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train { traj, noise, delay, timesteps, out } => {
            print_banner(&format!("TRAIN MODE - {}", traj.to_uppercase()));
            let args = vec![
                "--traj".to_string(), traj.clone(),
                "--noise".to_string(), noise.to_string(),
                "--delay".to_string(), delay.to_string(),
                "--timesteps".to_string(), timesteps.to_string(),
                "--out".to_string(), out.clone(),
            ];
            run_python("train.py", args);
        }
        Commands::Eval { model, traj, out, episodes } => {
            print_banner(&format!("EVAL MODE - {}", traj.to_uppercase()));
            let args = vec![
                "--model".to_string(), model.clone(),
                "--traj".to_string(), traj.clone(),
                "--out".to_string(), out.clone(),
                "--ep".to_string(), episodes.to_string(),
            ];
            run_python("evaluate.py", args);
        }
        Commands::RunAll { timesteps } => {
            print_banner("RUN ALL MODE");
            let trajectories = vec!["circle", "figure8", "lissajous"];
            
            for traj in trajectories {
                println!("\n>>> Processing Trajectory: {} <<<\n", traj);
                
                let train_args = vec![
                    "--traj".to_string(), traj.to_string(),
                    "--timesteps".to_string(), timesteps.to_string(),
                ];
                run_python("train.py", train_args);
                
                let eval_args = vec![
                    "--model".to_string(), format!("models/sac_final_{}", traj),
                    "--traj".to_string(), traj.to_string(),
                ];
                run_python("evaluate.py", eval_args);
            }
        }
    }
}
