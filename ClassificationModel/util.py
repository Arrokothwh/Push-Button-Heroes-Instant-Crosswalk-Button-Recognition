import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy
from scipy.signal import savgol_filter

dtype=torch.float32

# check acc 
def check_accuracy_final(loader, model, device, out=False):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype).unsqueeze(1)
            scores = model(x)
            preds = (torch.sigmoid(scores) > 0.5).long()

            # print prediction
            if out:
                pred_probs = torch.sigmoid(scores).flatten().tolist()
                labels = y.flatten().tolist()

                pred_str = ', '.join([f"{p * 100:.1f}%" for p in pred_probs])
                label_str = ', '.join(map(str, labels))

                print(f"Sample preds : [{pred_str}]")
                print(f"Sample labels: [{label_str}]")
            
            num_correct += (preds == y.long()).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        return acc
    
# train method
def train(model, optimizer, loader_train, loader_val, device, epochs=10, dtype=torch.float32):
    x1 = list(range(len(loader_train) * epochs))
    y1 = []
    y2 = []
    y3 = []
    model = model.to(device=device)
    criterion = nn.BCEWithLogitsLoss()
    cnt = 1
    total_cnt = epochs * len(loader_train)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype).unsqueeze(1)

            scores = model(x)
            loss = criterion(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y1.append(loss.item())
            if (cnt % 2 == 0):
                acc_val = check_accuracy_final(loader_val, model, device, out=True)
            else:
                acc_val = check_accuracy_final(loader_val, model, device)
            acc_train = check_accuracy_final(loader_train, model, device)
            y2.append(acc_val)
            y3.append(acc_train)
            print(f"Iter: {cnt}/{total_cnt:<5} |  Loss: {loss.item():<9.6f} |  Train Acc: {acc_train:<7.4f} |  Val Acc: {acc_val:<7.4f}")
            cnt += 1
    return (x1, y1, y2, y3)

# draw it vs train loss + it vs val acc & train acc
def plotpic(data, use_savgol=True, smooth_window=7):
    x, y1, y2, y3 = data

    if use_savgol:
        y1_smooth = savgol_filter(y1, window_length=smooth_window, polyorder=2)
        y2_smooth = savgol_filter(y2, window_length=smooth_window, polyorder=2)
        y3_smooth = savgol_filter(y3, window_length=smooth_window, polyorder=2)
    else:
        def smooth(y, w):
            kernel = np.ones(w) / w
            return np.convolve(y, kernel, mode='same')
        y1_smooth = smooth(y1, smooth_window)
        y2_smooth = smooth(y2, smooth_window)
        y3_smooth = smooth(y3, smooth_window)

    # Plot Loss
    plt.figure()
    plt.plot(x, y1_smooth, marker='.', label='Loss (smoothed)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.grid(True)

    # Plot Accuracy
    plt.figure()
    plt.plot(x, y2_smooth, marker='o', label='Validation Accuracy (smoothed)')
    plt.plot(x, y3_smooth, marker='o', label='Training Accuracy (smoothed)')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

# choosing hyperparameters
def hyperparameter_search(model_class, arch, loader_train, loader_val, device, epochs=5):
    lr_list = 10 ** np.random.uniform(-2, -6, (10,))
    wd_list = 10 ** np.random.uniform(-2, -4, (5,))
    beta_list = [(0.85, 0.99), (0.85, 0.995), (0.9, 0.98), (0.9, 0.99), (0.9, 0.995), (0.9, 0.999),
    (0.95, 0.98), (0.95, 0.99), (0.95, 0.995), (0.95, 0.999)]

    best_acc = 0.0
    best_model = None
    best_params = {}
    
    for lr, wd, betas in itertools.product(lr_list, wd_list, beta_list):
        print(f"\n Testing AdamW: lr={lr}, wd={wd}, betas={betas}")
        
        model = model_class(*arch).to(device)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": lr / 10},            
                {"params": model.classifier.parameters(), "lr": lr * 10},    
            ],
            weight_decay=wd,
            betas=betas
        )

        data = train(model, optimizer, loader_train, loader_val, device, epochs=epochs)
        _, _, val_accs, _ = data

        final_val_acc = val_accs[-1]
        print(f"Final val acc for lr={lr}, wd={wd}: {final_val_acc:.4f}")

        if final_val_acc > best_acc:
            best_acc = final_val_acc
            best_params = {"lr": lr, "weight_decay": wd}
            best_model = deepcopy(model)
            best_training_data = data

    print("\n Best Hyperparameters:")
    print(best_params)
    print(f"Best val acc: {best_acc:.4f}")

    # save model
    torch.save(best_model.state_dict(), "best_dino_model.pt")
    print(" Saved best model to best_dino_model.pt")

    return best_model, best_params, best_training_data

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
    plt.show()