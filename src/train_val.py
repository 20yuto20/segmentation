import torch
import numpy as np
import matplotlib.pyplot as plt

def train(cfg, device, model, train_progress_bar, optimizer, criterion, epoch):
    model.train()
    n_train = 0
    sum_loss = 0.0
    losses = []

    for i, sample in enumerate(train_progress_bar):
        image, label = sample['image'].to(device), sample['label'].to(device)

        if label.dim() == 4:
            label = label.squeeze(1)
        
        y = model(image)
        loss = criterion(y, label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        n_train += image.size(0)
        train_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        losses.append(loss.item())

        # Randomly sample and visualize predictions (every 100 iterations)
        if i % 100 == 0:
            with torch.no_grad():
                pred = torch.argmax(y, dim=1)
                for j in range(min(3, image.shape[0])):  # Visualize up to 3 samples
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    ax1.imshow(image[j].cpu().permute(1, 2, 0))
                    ax1.set_title("Input Image")
                    ax2.imshow(label[j].cpu())
                    ax2.set_title("True Label")
                    ax3.imshow(pred[j].cpu())
                    ax3.set_title("Prediction")
                    plt.savefig(f"{cfg.out_dir}debug_sample_epoch{epoch}_iter{i}_sample{j}.png")
                    plt.close()

    # After the training loop
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"{cfg.out_dir}training_loss_epoch{epoch}.png")
    plt.close()

    return sum_loss / n_train

def val(device, model, val_progress_bar, criterion, evaluator):
    model.eval()
    for sample in val_progress_bar:
        image, label = sample['image'].to(device), sample['label'].to(device)

        if label.dim() == 4:
            label = label.squeeze(1)

        with torch.no_grad():
            y = model(image)

        loss = criterion(y, label.long())
        pred = torch.argmax(y, dim=1)
        pred = pred.data.cpu().numpy()
        label = label.cpu().numpy()
        evaluator.add_batch(label, pred)
        val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()

    return mIoU, Acc

def test(cfg, device, model, test_loader, criterion, evaluator):
    model.eval()
    evaluator.reset()
    
    test_progress_bar = tqdm.tqdm(test_loader, desc='Testing')
    
    for sample in test_progress_bar:
        image, label = sample['image'].to(device), sample['label'].to(device)

        if label.dim() == 4:
            label = label.squeeze(1)

        with torch.no_grad():
            output = model(image)

        loss = criterion(output, label.long())
        pred = torch.argmax(output, dim=1)
        pred = pred.data.cpu().numpy()
        label = label.cpu().numpy()
        evaluator.add_batch(label, pred)
        test_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()
    
    print(f"Test Results - Accuracy: {Acc:.4f}, mIoU: {mIoU:.4f}")
    return mIoU, Acc