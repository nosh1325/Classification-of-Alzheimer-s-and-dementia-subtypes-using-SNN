!pip install snntorch

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import snntorch as snn
from snntorch import spikegen
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!pip install snntorch torchvision matplotlib

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, 
                           recall_score, accuracy_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import snntorch as snn
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 32
IMG_SIZE = 224
CNN_EPOCHS = 5
SNN_EPOCHS = 2
CNN_LR = 0.0001
SNN_LR = 0.005
FEATURE_DIM = 512
SNN_TIMESTEPS = 25

def load_dataset(path, label_map, is_train=True):
    images, labels = [], []
    if is_train:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, file)
                    if 'NonDemented' in root or 'nondemented' in root.lower():
                        images.append(img_path)
                        labels.append(0)
                    elif 'Demented' in root or 'demented' in root.lower():
                        images.append(img_path)
                        labels.append(1)
    else:
        for class_name in label_map.keys():
            class_path = os.path.join(path, class_name)
            if os.path.exists(class_path):
                for root, dirs, files in os.walk(class_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            images.append(os.path.join(root, file))
                            labels.append(label_map[class_name])
    return images, labels

class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='gray')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class CNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512, num_classes=2):
        super(CNNFeatureExtractor, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        x = self.backbone(x)
        features = self.feature_extractor(x)
        if return_features:
            return features
        out = self.classifier(features)
        return out, features

class SNNClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2, num_steps=25):
        super(SNNClassifier, self).__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1.0)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.lif2 = snn.Leaky(beta=0.85, threshold=1.0)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.lif3 = snn.Leaky(beta=0.8, threshold=1.0)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk3)
        return torch.stack(spk_rec).sum(0)

class SNNLatentExtractor(nn.Module):
    def __init__(self, snn_model):
        super(SNNLatentExtractor, self).__init__()
        self.snn = snn_model
        
    def extract_spike_features(self, spikes_over_time):
        timesteps, batch_size, features = spikes_over_time.shape
        feature_list = []
        feature_list.extend([
            spikes_over_time.mean(dim=0),
            spikes_over_time.sum(dim=0),
            spikes_over_time.std(dim=0),
            spikes_over_time.max(dim=0)[0]
        ])
        first_spike_time = torch.zeros(batch_size, features, device=spikes_over_time.device)
        for b in range(batch_size):
            for f in range(features):
                spike_times = torch.nonzero(spikes_over_time[:, b, f])
                first_spike_time[b, f] = spike_times[0].float() / timesteps if len(spike_times) > 0 else 1.0
        feature_list.append(first_spike_time)
        burst_count = torch.zeros(batch_size, features, device=spikes_over_time.device)
        for t in range(1, timesteps):
            burst_count += spikes_over_time[t-1] * spikes_over_time[t]
        feature_list.append(burst_count)
        feature_list.extend([
            spikes_over_time[:timesteps//3].mean(dim=0),
            spikes_over_time[timesteps//3:2*timesteps//3].mean(dim=0),
            spikes_over_time[2*timesteps//3:].mean(dim=0)
        ])
        z_t = torch.cat(feature_list, dim=1)
        return torch.clamp(torch.nan_to_num(z_t, 0.0), -10, 10)
    
    def encode_to_zt(self, cnn_features):
        self.snn.eval()
        with torch.no_grad():
            mem1 = self.snn.lif1.init_leaky()
            mem2 = self.snn.lif2.init_leaky()
            mem3 = self.snn.lif3.init_leaky()
            spk2_rec = []
            for step in range(self.snn.num_steps):
                cur1 = self.snn.fc1(cnn_features)
                spk1, mem1 = self.snn.lif1(cur1, mem1)
                cur2 = self.snn.fc2(spk1)
                spk2, mem2 = self.snn.lif2(cur2, mem2)
                cur3 = self.snn.fc3(spk2)
                spk3, mem3 = self.snn.lif3(cur3, mem3)
                spk2_rec.append(spk2)
            z_t = self.extract_spike_features(torch.stack(spk2_rec))
        return z_t.detach()
    
    def get_zt_batch(self, cnn_model, dataloader):
        z_t_vectors, labels_list = [], []
        cnn_model.eval()
        self.snn.eval()
        with torch.no_grad():
            for data, labels in tqdm(dataloader, desc='Extracting Z_T'):
                data = data.to(device)
                cnn_features = cnn_model(data, return_features=True)
                z_t = self.encode_to_zt(cnn_features)
                z_t_vectors.append(z_t.cpu())
                labels_list.append(labels)
        return torch.cat(z_t_vectors, dim=0), torch.cat(labels_list, dim=0)

def extract_zt_latent_variables(cnn_model, snn_model, train_loader, test_loader, save_path=None):
    zt_extractor = SNNLatentExtractor(snn_model)
    single_data = next(iter(train_loader))[0][:1].to(device)
    with torch.no_grad():
        cnn_features = cnn_model(single_data, return_features=True)
        test_zt = zt_extractor.encode_to_zt(cnn_features)
    if test_zt.std().item() < 1e-7:
        with torch.no_grad():
            for param in snn_model.parameters():
                param.data += torch.randn_like(param) * 0.1
    train_zt, train_labels = zt_extractor.get_zt_batch(cnn_model, train_loader)
    test_zt, test_labels = zt_extractor.get_zt_batch(cnn_model, test_loader)
    zt_data = {
        'train_zt': train_zt,
        'train_labels': train_labels,
        'test_zt': test_zt,
        'test_labels': test_labels,
        'latent_dim': train_zt.shape[1],
        'success': train_zt.std().item() > 1e-6
    }
    if save_path:
        torch.save(zt_data, save_path)
    return zt_data

def train_cnn(model, train_loader, val_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    best_val_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f'CNN Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_losses.append(running_loss/len(train_loader))
        train_accs.append(train_acc)
        val_acc = evaluate_cnn(model, val_loader)
        val_accs.append(val_acc)
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cnn_model.pth')
    
    return train_losses, train_accs, val_accs

def evaluate_cnn(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def extract_cnn_features(model, loader):
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Extracting CNN features'):
            images = images.to(device)
            features = model(images, return_features=True)
            features_list.append(features.cpu())
            labels_list.append(labels)
    return torch.cat(features_list, 0), torch.cat(labels_list, 0)

def train_snn(model, train_features, train_labels, val_features, val_labels, epochs, lr):
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, train_accs, val_accs = [], [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for features, labels in tqdm(train_loader, desc=f'SNN Epoch {epoch+1}/{epochs}'):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_losses.append(running_loss/len(train_loader))
        train_accs.append(train_acc)
        val_acc = evaluate_snn(model, val_features, val_labels)
        val_accs.append(val_acc)
    
    return train_losses, train_accs, val_accs

def evaluate_snn(model, features, labels):
    model.eval()
    all_preds = []
    for i in range(0, features.size(0), 64):
        batch_features = features[i:i+64].to(device)
        with torch.no_grad():
            outputs = model(batch_features)
            _, predicted = outputs.max(1)
        all_preds.append(predicted.cpu())
    all_preds = torch.cat(all_preds)
    return 100. * all_preds.eq(labels).sum().item() / features.size(0)

def evaluate_detailed(model, loader, model_type='cnn'):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'Evaluating {model_type}'):
            images = images.to(device)
            if model_type == 'cnn':
                outputs, _ = model(images)
            else:
                outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels)
            all_probs.append(probabilities[:, 1].cpu())
    
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def train_ensemble(train_features, train_labels, test_features, test_labels):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features.numpy())
    X_test = scaler.transform(test_features.numpy())
    y_train = train_labels.numpy()
    y_test = test_labels.numpy()
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    ensemble = VotingClassifier([('rf', rf), ('gb', gb)], voting='soft')
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    return ensemble, accuracy_score(y_test, y_pred)

def plot_training_history(cnn_losses, cnn_train_accs, cnn_val_accs, snn_losses, snn_train_accs, snn_val_accs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(cnn_losses, 'b-')
    axes[0, 0].set_title('CNN Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(cnn_train_accs, 'b-', label='Train')
    axes[0, 1].plot(cnn_val_accs, 'r-', label='Val')
    axes[0, 1].set_title('CNN Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(snn_losses, 'g-')
    axes[1, 0].set_title('SNN Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(snn_train_accs, 'g-', label='Train')
    axes[1, 1].plot(snn_val_accs, 'orange', label='Val')
    axes[1, 1].set_title('SNN Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Demented', 'Demented'],
                yticklabels=['Non-Demented', 'Demented'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_prob, auc_score):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(cnn_metrics, snn_metrics, ensemble_acc):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_values = [cnn_metrics['accuracy'], cnn_metrics['precision'], cnn_metrics['recall'], cnn_metrics['f1']]
    snn_values = [snn_metrics['accuracy'], snn_metrics['precision'], snn_metrics['recall'], snn_metrics['f1']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, cnn_values, width, label='CNN', color='steelblue')
    ax.bar(x + width/2, snn_values, width, label='SNN', color='coral')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(cnn_values):
        ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    for i, v in enumerate(snn_values):
        ax.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_zt_distribution(zt_data, save_path='zt_distribution.png'):
    train_zt = zt_data['train_zt']
    train_labels = zt_data['train_labels']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(train_zt.flatten().numpy(), bins=50, alpha=0.7, color='purple')
    axes[0, 0].set_title('Z_T Distribution')
    axes[0, 0].set_xlabel('Z_T Values')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    mean_per_dim = train_zt.mean(dim=0).numpy()
    axes[0, 1].plot(mean_per_dim[:100], color='red', linewidth=2)
    axes[0, 1].set_title('Mean Activation (First 100 dims)')
    axes[0, 1].set_xlabel('Z_T Dimension')
    axes[0, 1].set_ylabel('Mean Activation')
    axes[0, 1].grid(True, alpha=0.3)
    
    if train_zt.shape[1] >= 2:
        for label in torch.unique(train_labels):
            mask = train_labels == label
            zt_class = train_zt[mask]
            label_name = 'Non-Demented' if label.item() == 0 else 'Demented'
            axes[1, 0].scatter(zt_class[:, 0], zt_class[:, 1], alpha=0.6, label=label_name, s=30)
        axes[1, 0].set_title('Z_T Space (2D)')
        axes[1, 0].set_xlabel('Z_T Dim 0')
        axes[1, 0].set_ylabel('Z_T Dim 1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    sample_size = min(100, train_zt.shape[0])
    sample_dims = min(50, train_zt.shape[1])
    heatmap_data = train_zt[:sample_size, :sample_dims].numpy()
    sns.heatmap(heatmap_data.T, cmap='viridis', cbar=True, ax=axes[1, 1])
    axes[1, 1].set_title('Z_T Heatmap')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Z_T Dimension')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    train_path = '/kaggle/input/alzheimerdataset'  
    test_path = '/kaggle/input/dementia-nins-v1'   
    label_map = {'NonDemented': 0, 'Demented': 1}
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_images, train_labels = load_dataset(train_path, label_map, is_train=True)
    test_images, test_labels = load_dataset(test_path, label_map, is_train=False)
    
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")
    print(f"Train distribution: {Counter(train_labels)}")
    print(f"Test distribution: {Counter(test_labels)}\n")
    
    train_dataset = BrainMRIDataset(train_images, train_labels, train_transform)
    test_dataset = BrainMRIDataset(test_images, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print("Training CNN")
    cnn_model = CNNFeatureExtractor(feature_dim=FEATURE_DIM, num_classes=2).to(device)
    cnn_losses, cnn_train_accs, cnn_val_accs = train_cnn(cnn_model, train_loader, test_loader, CNN_EPOCHS, CNN_LR)
    cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
    cnn_metrics = evaluate_detailed(cnn_model, test_loader, model_type='cnn')
    
    print("Extracting CNN features...")
    train_features, train_labels_tensor = extract_cnn_features(cnn_model, train_loader)
    test_features, test_labels_tensor = extract_cnn_features(cnn_model, test_loader)
    print(f"Train features: {train_features.shape}, Test features: {test_features.shape}\n")
    
    print("Training SNN")
    snn_model = SNNClassifier(input_dim=FEATURE_DIM, hidden_dim=256, num_classes=2, num_steps=SNN_TIMESTEPS).to(device)
    snn_losses, snn_train_accs, snn_val_accs = train_snn(snn_model, train_features, train_labels_tensor, test_features, test_labels_tensor, SNN_EPOCHS, SNN_LR)
    
    all_snn_preds, all_snn_probs = [], []
    for i in range(0, test_features.size(0), BATCH_SIZE):
        batch_feat = test_features[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            outputs = snn_model(batch_feat)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
        all_snn_preds.append(preds.cpu())
        all_snn_probs.append(probs[:, 1].cpu())
    
    all_snn_preds = torch.cat(all_snn_preds).numpy()
    all_snn_probs = torch.cat(all_snn_probs).numpy()
    
    snn_metrics = {
        'accuracy': accuracy_score(test_labels_tensor.numpy(), all_snn_preds),
        'precision': precision_score(test_labels_tensor.numpy(), all_snn_preds, average='binary', zero_division=0),
        'recall': recall_score(test_labels_tensor.numpy(), all_snn_preds, average='binary', zero_division=0),
        'f1': f1_score(test_labels_tensor.numpy(), all_snn_preds, average='binary', zero_division=0),
        'auc': roc_auc_score(test_labels_tensor.numpy(), all_snn_probs),
        'confusion_matrix': confusion_matrix(test_labels_tensor.numpy(), all_snn_preds),
        'y_true': test_labels_tensor.numpy(),
        'y_pred': all_snn_preds,
        'y_prob': all_snn_probs
    }
    
    print("Extracting Z_T latent variables")
    zt_data = extract_zt_latent_variables(cnn_model, snn_model, train_loader, test_loader, save_path='zt_latent_variables.pt')
    print(f"Z_T shape: {zt_data['train_zt'].shape}, Latent dim: {zt_data['latent_dim']}\n")
    
    print("Training ensemble on Z_T")
    ensemble_model, ensemble_acc = train_ensemble(zt_data['train_zt'], zt_data['train_labels'], zt_data['test_zt'], zt_data['test_labels'])
    
    
    plot_training_history(cnn_losses, cnn_train_accs, cnn_val_accs, snn_losses, snn_train_accs, snn_val_accs)
    plot_confusion_matrix(cnn_metrics['confusion_matrix'], 'CNN Confusion Matrix')
    plot_confusion_matrix(snn_metrics['confusion_matrix'], 'SNN Confusion Matrix')
    plot_roc_curve(cnn_metrics['y_true'], cnn_metrics['y_prob'], cnn_metrics['auc'])
    plot_metrics_comparison(cnn_metrics, snn_metrics, ensemble_acc)
    visualize_zt_distribution(zt_data, save_path='zt_distribution.png')
    
    print("Results:")
    print(f"CNN Test - Acc: {cnn_metrics['accuracy']:.4f}, F1: {cnn_metrics['f1']:.4f}, AUC: {cnn_metrics['auc']:.4f}")
    print(f"SNN Test - Acc: {snn_metrics['accuracy']:.4f}, F1: {snn_metrics['f1']:.4f}, AUC: {snn_metrics['auc']:.4f}")
    print(f"Ensemble (Z_T) - Acc: {ensemble_acc:.4f}")
    print(f"\nCNN Confusion Matrix:\n{cnn_metrics['confusion_matrix']}")
    print(f"SNN Confusion Matrix:\n{snn_metrics['confusion_matrix']}")


if __name__ == '__main__':
    main()