import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import shutil

class Urban_Rural_Uninhabited_PretrainingDataset(Dataset):
    def __init__(self, labels_df, image_folder, transform=None):
        self.labels_df = labels_df
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_id = row['Image ID']
        image_path = os.path.join(self.image_folder, f"{image_id}.png")
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor([row['Urban'], row['Rural'], row['Mountain'], row['Highland']], dtype=torch.float32)
        label = label / label.sum()
        # label smoothing
        epsilon = 1e-7
        label = (1 - epsilon) * label + epsilon /4
        return image, label

class ResNet18(nn.Module):
    def __init__(self, pretrain=False):
        super(ResNet18, self).__init__()
        if pretrain:
            print('Use pretrained ResNet18')
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            print('Use non-pretrained ResNet18')
            self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4) # 修改全连接层以适应4分类
    def forward(self, x):
        x = self.model(x)
        return x

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, patience):
    best_val_acc, best_val_loss, epochs_no_improve = 0.0, float('inf'), 0

    for epoch in range(num_epochs):
        logging.info(f"Start training epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss, correct, total = 0.0, 0, 0
        model.eval()
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}")
            for images, labels in val_iter:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                
                pred_labels = preds.argmax(dim=1)
                true_labels = labels.argmax(dim=1)
                correct += (pred_labels == true_labels).sum().item()

                total += labels.size(0)

        val_acc = correct / total

        scheduler.step()
        # get learning rate and print it in logging information
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, "
                     f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Current Learning Rate: {lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saving best model, validation accuracy improved to {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Early stopping patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Validation loss has not improved for {patience} epochs, early stopping.")
            break

def softmax_mse_loss(preds, labels):
    preds = F.softmax(preds, dim=1)
    num_classes = preds.size(1)
    return F.mse_loss(preds, labels, reduction='sum') / num_classes

if __name__ == "__main__":
    image_folder = "../data/pretrain_data_small"
    test_image_folder = "../data/48RVV_32_times_32"           # just for testing, need to replace for complete data later
    label_file = "../data/pretrain_labels_small.csv"
    classification_dir = '../data/classified_images'
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.0001
    model_save_path = "pretrained_resnet18.pth"
    patience = 30
    transform = transforms.Compose([
            transforms.Resize((224, 224)),                
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(45),
            #transforms.RandomGrayscale(p=0.1),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # check if the trained model exists
    if os.path.exists(model_save_path):
        # load model
        model = ResNet18()
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model.eval()
        print("Model loaded successfully")
        # traverse test_image_folder and make predictions
        class_names = ['Urban', 'Rural', 'Mountain', 'Highland']
        os.makedirs(classification_dir, exist_ok=True)
        for class_name in class_names:
            os.makedirs(os.path.join(classification_dir, class_name), exist_ok=True)
        test_files = os.listdir(test_image_folder)
        for file in test_files:
            image_path = os.path.join(test_image_folder, file)
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image)

            # get the predicted class
            choice = output.argmax(dim=1).item()
            probs = F.softmax(output, dim=1).squeeze().tolist()
            print(f"Image: {file}, Distribution: {probs}, Prediction: {choice}")
            
            predicted_class = class_names[choice]
            dest_path = os.path.join(classification_dir, predicted_class, file)
            shutil.copy(image_path, dest_path)
    else:
        print("Start training the model")
        labels_df = pd.read_csv(label_file)
        train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

        train_dataset = Urban_Rural_Uninhabited_PretrainingDataset(train_df, image_folder, transform=transform)
        val_dataset = Urban_Rural_Uninhabited_PretrainingDataset(val_df, image_folder, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = ResNet18(pretrain=True)

        # distributed training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # n_gpus = torch.cuda.device_count()
        # if n_gpus > 1:
        #     model = nn.DataParallel(model)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, patience)

        print(f"Best model saved to {model_save_path}")