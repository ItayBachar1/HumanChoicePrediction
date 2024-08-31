import wandb

YOUR_WANDB_USERNAME = "itaybachartechnion"
project = "NLP2024_PROJECT_204117071_206948218"

command = [
    "${ENVIRONMENT_VARIABLE}",
    "${interpreter}",
    "StrategyTransfer.py",
    "${project}",
    "${args}"
]

sweep_config = {
    "name": "sweep Random,history_Llama3,Llama3",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "seed": {"values": list(range(1, 6))},
        "features": {"values": ["EFs"]},
        "basic_nature": {'values': [25]},
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")