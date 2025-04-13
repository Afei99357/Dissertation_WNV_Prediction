import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set paths relative to script location
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(__file__).resolve().parents[2] / "data_demo" / "cdc_demo_with_climate.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Preprocess dataset
data["Population"] = data["Population"].str.replace(",", "").str.strip()
data["Population"] = pd.to_numeric(data["Population"], errors="coerce")

# Select relevant columns
date_index = data.columns.get_loc("Date")
data_pred = data.iloc[:, date_index + 1:].copy()
data_pred["Year"] = data["Year"]
data_pred["Reported human cases"] = data["Reported human cases"]  # check spelling
data_pred["FIPS"] = data["FIPS"]

# Split into train and test sets
train = data_pred[data_pred["Year"] < 2018]
test = data_pred[data_pred["Year"] >= 2018]

# Impute missing human case values using FIPS-wise mean from training data
fips_avg = train.groupby("FIPS")["Reported human cases"].mean()
train["Reported human cases"] = train["Reported human cases"].fillna(train["FIPS"].map(fips_avg))
test["Reported human cases"] = test["Reported human cases"].fillna(test["FIPS"].map(fips_avg))

# Remove rows with any remaining NaN
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

# Separate labels from features
train_labels = train.pop("Reported human cases").values
test_labels = test.pop("Reported human cases").values

# Drop year column
train = train.drop(columns=["Year"])
test = test.drop(columns=["Year"])

# Convert data to PyTorch tensors
train_features = torch.tensor(train.values, dtype=torch.float32)
test_features = torch.tensor(test.values, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)
test_labels = torch.tensor(test_labels, dtype=torch.float32).view(-1, 1)

# Define a simple neural network for regression
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.SELU(),
        )

    def forward(self, x):
        return self.layers(x)

# Initialize model, optimizer, and loss
input_size = train_features.shape[1]
model = NeuralNetwork(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Custom R2 score metric
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-6)

# Training loop
epochs = 100
train_losses, val_losses = [], []
train_r2_scores, val_r2_scores = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(train_features)
    loss = loss_fn(predictions, train_labels)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        val_preds = model(test_features)
        val_loss = loss_fn(val_preds, test_labels).item()
        train_r2 = r2_score(train_labels, predictions).item()
        val_r2 = r2_score(test_labels, val_preds).item()

    train_losses.append(loss.item())
    val_losses.append(val_loss)
    train_r2_scores.append(train_r2)
    val_r2_scores.append(val_r2)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

# Plot training loss
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.yscale("log")
plt.title("Loss Over Epochs")
plt.savefig(RESULTS_DIR / "loss_plot.png")
plt.show()

# Plot R2 scores
plt.plot(train_r2_scores, label="Train R2")
plt.plot(val_r2_scores, label="Validation R2")
plt.legend()
plt.title("R2 Score Over Epochs")
plt.savefig(RESULTS_DIR / "r2_plot.png")
plt.show()

# Evaluate model on test set
model.eval()
with torch.no_grad():
    test_predictions = model(test_features)
    test_r2 = r2_score(test_labels, test_predictions).item()
    test_mse = loss_fn(test_predictions, test_labels).item()

print(f"Test R2 Score: {test_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Save predictions to CSV
results = pd.DataFrame({
    "test_labels": test_labels.numpy().flatten(),
    "predictions": test_predictions.numpy().flatten()
})
results.to_csv(RESULTS_DIR / "NN_results.csv", index=False)