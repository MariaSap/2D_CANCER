import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import glob
from PIL import Image
import time

# ==========================================
# 1. Custom Dataset to Handle Medical Folders
# ==========================================
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []

        # Robust file finding
        print(f"Scanning {root_dir}...")
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            # Recursively find all images (handling subfolders and different extensions)
            image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.PNG', '*.JPG']:
                # recursive=True allows finding images inside sub-folders (e.g., patient folders)
                image_paths.extend(glob.glob(os.path.join(class_dir, '**', ext), recursive=True))
            
            print(f"  > Class '{class_name}': Found {len(image_paths)} images")
            
            for img_path in image_paths:
                self.samples.append((img_path, self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 files in {root_dir}. Please check your GAN output path.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        
        # Load as Grayscale (L) for medical images
        try:
            sample = Image.open(path).convert('L') 
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a black image in case of error to prevent crash
            sample = Image.new('L', (256, 256))

        if self.transform:
            sample = self.transform(sample)

        return sample, target

# ==========================================
# 2. Updated TSTR Evaluator
# ==========================================
class TSTREvaluator:
    def __init__(self, device='cuda', img_size=(256, 256), batch_size=32, epochs=10):
        self.device = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])

    def _get_model(self, num_classes):
        model = models.resnet18(weights=None)
        # Modify first layer for 1-channel input (Grayscale)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify last layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model.to(self.device)

    def run_tstr(self, synthetic_data_path, real_test_data_path):
        print(f"\n{'='*20} Starting TSTR Evaluation (Robust Mode) {'='*20}")
        
        # --- CHANGED: Use Custom MedicalImageDataset instead of ImageFolder ---
        try:
            train_dataset = MedicalImageDataset(synthetic_data_path, transform=self.transform)
            test_dataset = MedicalImageDataset(real_test_data_path, transform=self.transform)
        except RuntimeError as e:
            print(f"CRITICAL ERROR: {e}")
            return 0.0

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0) # num_workers=0 is safer on Windows
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        class_names = train_dataset.classes
        num_classes = len(class_names)
        
        print(f"Total Training (Synthetic): {len(train_dataset)}")
        print(f"Total Testing (Real): {len(test_dataset)}")

        # Model Setup
        model = self._get_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        print(f"\nTraining classifier on SYNTHETIC data for {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Prevent division by zero if loader is empty
            avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
            epoch_acc = 100 * correct / total if total > 0 else 0
            print(f" Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | Synth Acc: {epoch_acc:.2f}%")

        print(f"Training finished in {time.time() - start_time:.1f}s")

        # Testing Loop
        print("\nTesting classifier on REAL data...")
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n{'='*20} TSTR RESULTS {'='*20}")
        print(f"Final Accuracy on Real Data: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Only print detailed report if we actually have predictions
        if len(all_labels) > 0:
            print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
            print("Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
        
        return accuracy

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Ensure these paths are exactly correct
    SYNTH_DIR = r'C:\Users\sapounaki.m\Desktop\2D_CANCER\data\synth'
    REAL_DIR = r'C:\Users\sapounaki.m\Desktop\2D_CANCER\data\train'  
    
    tstr = TSTREvaluator(device='cuda', epochs=20)
    # Run Evaluation
    try:
        score = tstr.run_tstr(SYNTH_DIR, REAL_DIR)
        
        # INTERPRETATION
        print("\nINTERPRETATION:")
        if score > 0.85:
            print(">> EXCELLENT: The synthetic data captures highly diagnostic features.")
        elif score > 0.70:
            print(">> GOOD: The GAN has learned general structures but might miss subtle pathology.")
        elif score > 0.50:
            print(">> WEAK: The classifier is struggling. GAN might be generating noisy or unrealistic textures.")
        else:
            print(">> FAILURE: The synthetic data does not resemble the real data distribution.")
            
    except Exception as e:
        print(f"An error occurred: {e}")