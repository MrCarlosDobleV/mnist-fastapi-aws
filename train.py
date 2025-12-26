import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=64,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=1000,
        shuffle=False,
    )

    model = MNISTNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(
            f"Epoch {epoch + 1} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_acc * 100:.2f}%"
        )

    torch.save(model.state_dict(), "mnist_model.pt")
    print("Model saved as mnist_model.pt")

if __name__ == "__main__":
    main()
