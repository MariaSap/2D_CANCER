# src/run.py
import os, torch
from data import build_loaders
from pathlib import Path
from classifier import make_classifier, train_classifier
from train_gan import train_gan
from sample_gan import sample_to_folder
from merge_data import merge_real_synth

DATA_ROOT = Path(r"C:data")
SYNTH_ROOT = Path(r"data\synth\train")
MERGED_ROOT = Path(r"data\merged_train")

def main():
    # 1) Baseline classifier on real
    train_dl, val_dl, test_dl, classes = build_loaders(DATA_ROOT, batch_size=32)
    clf = make_classifier(num_classes=len(classes))
    clf = train_classifier(clf, train_dl, val_dl, epochs=20, lr=1e-3, device="cpu")

    # 2) Train conditional GAN on real train set
    g_ema = train_gan(train_dl, num_classes=len(classes), iters=50000, device="cpu")

    # 3) Sample synthetic data (balance per class)
    if os.path.exists(SYNTH_ROOT) is False:
        os.makedirs(SYNTH_ROOT, exist_ok=True)
    sample_to_folder(g_ema, SYNTH_ROOT, per_class=1000, num_classes=len(classes), device="cpu")

    # 4) Merge real + synthetic for augmented training
    merge_real_synth(os.path.join(DATA_ROOT, "train"), SYNTH_ROOT, MERGED_ROOT)
    # reuse val/test from real-only
    # Build loaders for merged train
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from data import build_transforms
    merged_ds = datasets.ImageFolder(MERGED_ROOT, transform=build_transforms(True))
    merged_dl = DataLoader(merged_ds, batch_size=32, shuffle=True, num_workers=4)

    # 5) Train classifier on augmented data
    clf_aug = make_classifier(num_classes=len(classes))
    clf_aug = train_classifier(clf_aug, merged_dl, val_dl, epochs=20, lr=1e-3, device="cpu")

    # 6) Evaluate both on the same test set
    def eval_acc(model, dl):
        model.eval()
        device = "cpu"
        correct, total = 0, 0
        with torch.no_grad():
            for x,y in dl:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.numel()
        return correct/total

    acc_real = eval_acc(clf, test_dl)
    acc_aug  = eval_acc(clf_aug, test_dl)
    print("Test acc (real):", acc_real)
    print("Test acc (aug):", acc_aug)

if __name__ == "__main__":
    main()
