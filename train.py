from dataset import loader
from model import Net
import torch.nn as nn
import torch.optim as optim
import torch

net = Net()


weights = torch.tensor([
    1.0,   # Dur
    1.0,   # Ileri
    1.0,   # Geri
    1.0,   # Kalk
    1.1    # Unknown
])

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(
    net.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",      # loss should decrease
    factor=0.5,
    patience=15
)

epochs = 175
best_loss = float("inf")
for epoch in range(epochs):

    correct = 0
    total = 0
    running_loss = 0.0

    net.train()   # Put the model into training mode

    for inputs, labels in loader:

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    average_loss = running_loss / len(loader)

    scheduler.step(average_loss)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(net.state_dict(), "voice_model.pth")
        print("Best model saved!")
