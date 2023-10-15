import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from ray import tune
import wandb

def train_model(config):
    # Initialize Weights and Biases run
    wandb.init(project="my_project", config=config)

    # Your custom dataset
    full_dataset = text_dataset()
    
    # Splitting into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)
    
    # Define your model, loss function, and optimizer
    model = YourModelClass(config["param1"], config["param2"], ...)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_loader:
            inputs, labels, ... = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            wandb.log({"Train Loss": loss.item()})
        
        model.eval()
        val_loss = 0.0
        for batch in val_loader:
            with torch.no_grad():
                inputs, labels, ... = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"Validation Loss": avg_val_loss})

        # Pass the validation loss to Tune
        tune.report(val_loss=avg_val_loss)

# Hyperparameter tuning with Ray Tune
analysis = tune.run(
    train_model,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": 10,
        "param1": ...,
        "param2": ...,
        ...
    }
)

print("Best hyperparameters found were: ", analysis.best_config)




# Find the trial with the best end validation loss
best_trial = analysis.get_best_trial("val_loss", "min", "last")

# Load the best model
best_checkpoint_dir = best_trial.checkpoint.value
model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
model = YourModel()  # Initialize the model again
model.load_state_dict(model_state)



from ray.tune.integration.wandb import WandbLoggerCallback

wandb_init = {
    "project": "my_project",
    "group": "experiment_group",
    "job_type": "train",
}

config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([16, 32, 64]),
}

analysis = tune.run(
    train_model,
    config=config,
    callbacks=[WandbLoggerCallback(
        project=wandb_init["project"],
        group=wandb_init["group"],
        job_type=wandb_init["job_type"],
        sync_config=wandb_init,
    )],
)
