import copy
import torch
import matplotlib.pyplot as plt


# show prediction
def show_predictions_grid(loader, model, device, class_names=None, max_images=12, images_per_row=3):
    model.eval()
    images_shown = 0

    all_imgs = []
    all_titles = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            probs = torch.sigmoid(scores)
            preds = (probs > 0.5).long().squeeze(1)

            for i in range(x.size(0)):
                if images_shown >= max_images:
                    break

                img = x[i].cpu()
                np_img = img.permute(1, 2, 0).numpy()

                pred_label = preds[i].item()
                true_label = y[i].item()
                pred_text = class_names[pred_label] if class_names else str(pred_label)
                true_text = class_names[true_label] if class_names else str(true_label)
                title = f"Pred: {pred_text} / GT: {true_text}"

                all_imgs.append(np_img)
                all_titles.append(title)
                images_shown += 1
            if images_shown >= max_images:
                break

    num_cols = images_per_row
    num_rows = (len(all_imgs) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))

    for i, ax in enumerate(axes.flat):
        if i < len(all_imgs):
            ax.imshow(all_imgs[i])
            ax.set_title(all_titles[i], fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("prediction_grid.png")
    plt.show()


def evaluate(model, dataloader, criterion, device):
    """Return average loss and accuracy for a dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)  # binary‑class task

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


# -----------------------------------------------------------------------------
#                                 PLOTTING
# -----------------------------------------------------------------------------

def plot_accuracy_curves(
    train_acc: list[float],
    val_acc: list[float],
    train_loss: list[float] | None = None,
    val_loss: list[float] | None = None,
    model_name: str | None = "model",
):
    """Plot training curves.

    Always produces an **accuracy vs epoch** figure. If `train_loss` & `val_loss` are
    provided it also produces a second **loss vs epoch** figure.
    The plots are saved to PNG files in the working directory and shown to screen.
    """

    epochs = range(1, len(train_acc) + 1)

    # --------------------------- ACCURACY PLOT ---------------------------
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curves for {model_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"accuracy_curve_{model_name}.png")

    # ----------------------------- LOSS PLOT ----------------------------
    if train_loss is not None and val_loss is not None:
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        # plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves for {model_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"loss_curve_{model_name}.png")

    # Display figures (both shown consecutively if two exist)
    plt.show()


# -----------------------------------------------------------------------------
#                                   TRAINING
# -----------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs: int = 20,
    log_interval: int = 1,
    early_stopping_patience: int = 5,
    early_stopping_delta: float = 0.0,
):
    """Train the model with early stopping.

    Returns
    -------
    train_acc_history : list[float]
        Epoch‑wise training accuracy.
    val_acc_history   : list[float]
        Epoch‑wise validation accuracy.
    train_loss_history : list[float]
        Epoch‑wise average training loss.
    val_loss_history   : list[float]
        Epoch‑wise average validation loss.
    """

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    train_acc_history, val_acc_history = [], []
    train_loss_history, val_loss_history = [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # --------------------- TRAIN PHASE --------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # accuracy on train batch
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)

        # ------------------ VALIDATION PHASE ------------------
        epoch_val_loss, epoch_val_acc = evaluate(model, val_loader, criterion, device)
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        # -------------------- LOG RESULTS ---------------------
        if epoch % log_interval == 0:
            print(
                f"Train ‑ loss: {epoch_train_loss:.4f}, acc: {epoch_train_acc * 100:.2f}%\n"
                f"Val   ‑ loss: {epoch_val_loss:.4f}, acc: {epoch_val_acc * 100:.2f}%"
            )

        # ------------------ EARLY STOPPING --------------------
        if epoch_val_loss + early_stopping_delta < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"\nNo improvement in validation loss for {early_stopping_patience} "
                    "epochs – early stopping."
                )
                break

    # restore best model weights
    model.load_state_dict(best_model_wts)

    return train_acc_history, val_acc_history, train_loss_history, val_loss_history


__all__ = [
    "evaluate",
    "plot_accuracy_curves",
    "train",
]
