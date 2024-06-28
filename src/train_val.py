import tqdm

import torch

def train(device, model, train_progress_bar, optimizer, criterion):
    # ネットワークを学習モードへ変更
    model.train()

    sum_loss = 0.0
    for sample in train_progress_bar:
        image, label = sample['image'], sample['label']
        image = image.to(device)
        label = label.to(device)
        y = model(image)
        loss = criterion(y, label.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        train_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return sum_loss

def val(device, model, val_progress_bar, criterion, evaluator):
    # ネットワークを評価モードへ変更
    model.eval()
    # 評価の実行
    for sample in val_progress_bar:
        image, label = sample['image'], sample['label']
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            y = model(image)

        loss = criterion(y, label.long())
        sum_loss += loss.item()
        pred = torch.argmax(y, dim=1)
        pred = pred.data.cpu().numpy()
        label = label.cpu().numpy()
        evaluator.add_batch(label, pred)
        val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    mIoU = evaluator.Mean_Intersection_over_Union()
    Acc = evaluator.Pixel_Accuracy()

    return mIoU, Acc


    
    
def test():
    print(test)
