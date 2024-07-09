import tqdm

import torch

def train(device, model, train_progress_bar, optimizer, criterion):
    # ネットワークを学習モードへ変更
    model.train()

    n_train = 0
    sum_loss = 0.0
    for sample in train_progress_bar:
        image, label = sample['image'].to(device), sample['label'].to(device)

        # チャンネルの次元を削除する（損失関数が３次元を受け取っているから）
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

    return sum_loss / n_train

def val(device, model, val_progress_bar, criterion, evaluator):
    # ネットワークを評価モードへ変更
    model.eval()
    # 評価の実行
    for sample in val_progress_bar:
        image, label = sample['image'].to(device), sample['label'].to(device)

        # trainと同様に1次元削除して3次元へ
        if label.dim() == 4:
            label = label.squeeze(1)

        with torch.no_grad():
            y = model(image)

        loss = criterion(y, label.long())
        # sum_loss += loss.item()
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

        # 次元の削除
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
