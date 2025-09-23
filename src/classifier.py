import torch, torch.nn as nn, torch.optim as optim
from torchvision import models

def make_classifier(num_classes=4):
    model = models.resnet18(weights=None)
    # adjust first conv for 1-channel input
    w = model.conv1.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = w
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_classifier(model, train_dl, val_dl, epochs=25, lr=1e-3, device="cpu"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = {"acc": 0.0, "state":None}
    
    for ep in range(epochs):
        model.train()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        model.eval()
        correct, total = 0,0

        with torch.no_grad():
            for x,y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct +=(pred==y).sum().item()
                total += y.numel()

        acc = correct/total
        if acc > best["acc"]:
            best["acc"] = acc
            best["state"] = model.state_dict()
            
    model.load_state_dict(best["state"])
    return model