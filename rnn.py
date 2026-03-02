import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def generate_data(seq_length: int = 50, num_samples: int = 1000) -> tuple:
   
    X = np.linspace(0, 100, num_samples)
    y = np.sin(X)
    sequences = []
    targets = []

    for i in range(len(X) - seq_length):
        sequences.append(y[i:i + seq_length])
        targets.append(y[i + seq_length])

   
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, label='Sine Wave', color='blue', linewidth=2)
    plt.title('Sine Wave Data for Training', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return np.array(sequences), np.array(targets)


class RNN(nn.Module):
    
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


def train_model(model, train_loader, criterion, optimizer, epochs: int, device: str) -> list:
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
    
    return train_losses


def evaluate_and_visualize(model, seq_length: int, device: str) -> None:
    model.eval()
    
    X_extended = np.linspace(0, 150, 1000)
    y_extended = np.sin(X_extended)
    predictions = []
    indices = []
    
    for i in range(100, len(y_extended) - seq_length):
        window = y_extended[i:i + seq_length]
        X_window = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(X_window).detach().cpu().numpy()
        predictions.append(pred[0, 0])
        indices.append(i + seq_length)
    
    test_true = y_extended[indices]
    test_x = X_extended[indices]
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_x, test_true, label='True', color='blue', linewidth=2)
    plt.plot(test_x, predictions, label='Predicted', color='red', linestyle='--', linewidth=2)
    plt.legend()
    plt.title('RNN Predictions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    mse = np.mean((np.array(predictions) - test_true) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - test_true))
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Hyperparameters
    seq_length = 50
    input_size = 1
    hidden_size = 16
    output_size = 1
    num_layers = 1
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    
    # Generate and prepare data
    print("Generating training data...")
    X, y = generate_data(seq_length)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    # Initialize model
    model = RNN(input_size, hidden_size, output_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("Training model...")
    train_losses = train_model(model, train_loader, criterion, optimizer, epochs, device)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', 
             color='blue', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Test and visualize predictions
    print("Generating test data...")
    evaluate_and_visualize(model, seq_length, device)
    
    print("Done!")


if __name__ == "__main__":
    main()