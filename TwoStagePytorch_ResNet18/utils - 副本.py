import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# def train_one_epoch(model, dataloader, optimizer, criterion, device):
#     model.train()
#     running_loss = 0.0
#     for inputs, labels in dataloader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         labels = labels.float().unsqueeze(1)  # Step 2
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * inputs.size(0)
#     return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            labels = labels.float().unsqueeze(1)  # Step 2
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Step 3
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return running_loss / len(dataloader.dataset), accuracy


def plot_training_curves(loss_curve, acc_curve_x, acc_curve_y, model_name):
    plt.figure()
    plt.plot(loss_curve, label='Train Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve for {model_name}')
    plt.grid(True)
    plt.savefig(f'Training Loss Curve for {model_name}.png')

    plt.figure()
    plt.plot(acc_curve_x, acc_curve_y, marker='o', label='Val Accuracy', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy Curve for {model_name}')
    plt.grid(True)
    plt.savefig(f'Validation Accuracy Curve for {model_name}.png')
    plt.show()


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=5, log_interval=5):
    loss_curve = []
    acc_curve_x = []
    acc_curve_y = []

    total_steps = epochs * len(train_loader)
    step = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        model.train()
        running_loss = 0.0
        sample_count = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 记录 loss
            loss_curve.append(loss.item())

            # ✅ 累加用于 epoch 平均 loss
            running_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)

            # 每 log_interval 步打印 val acc
            if step % log_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                train_loss = running_loss / sample_count  # ✅ 计算当前平均训练损失
                acc_curve_x.append(step)
                acc_curve_y.append(val_acc)
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

            step += 1

    return loss_curve, acc_curve_x, acc_curve_y