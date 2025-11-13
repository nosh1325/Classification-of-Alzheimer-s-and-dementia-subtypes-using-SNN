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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, 
                           precision_score, recall_score, accuracy_score)
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TIME_STEPS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.008
EPOCHS = 10  
IMG_SIZE = 32
INPUT_SIZE = IMG_SIZE * IMG_SIZE
FAST_TRAINING_BATCHES = 20
FAST_TRAIN_FEATURES = 50

print(f"Training batches: {FAST_TRAINING_BATCHES}, Feature batches: {FAST_TRAIN_FEATURES}, Epochs: {EPOCHS}")

images = []
labels = []

def load_datasets(original_path, additional_path=None):
    images = []
    labels = []
    
    for subfolder in os.listdir(original_path):
        subfolder_path = os.path.join(original_path, subfolder)
        for folder in os.listdir(subfolder_path):
            subfolder_path2 = os.path.join(subfolder_path, folder)
            for image_filename in os.listdir(subfolder_path2):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(subfolder_path2, image_filename)
                    images.append(image_path)
                    labels.append(folder)
    
    if additional_path and os.path.exists(additional_path):
        for class_folder in ['Demented', 'NonDemented']:
            class_path = os.path.join(additional_path, class_folder)
            if os.path.exists(class_path):
                items = os.listdir(class_path)
                for item in items:
                    item_path = os.path.join(class_path, item)
                    if os.path.isdir(item_path):
                        for image_file in os.listdir(item_path):
                            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                image_path = os.path.join(item_path, image_file)
                                images.append(image_path)
                                labels.append(class_folder)
                    elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        images.append(item_path)
                        labels.append(class_folder)
    
    return images, labels

BASE_PATH = '/kaggle/input/alzheimerdataset'
ADDITIONAL_PATH = '/kaggle/input/nins-dementia-v2'

images, labels = load_datasets(BASE_PATH, ADDITIONAL_PATH)
df = pd.DataFrame({'image': images, 'label': labels})

def create_binary_labels(label):
    return 0 if label == "NonDemented" else 1

df['label_idx'] = df['label'].apply(create_binary_labels)
label_to_idx = {'NonDemented': 0, 'Demented': 1}

print(f"Total samples: {len(df)}")

class AlzheimerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image']
        label = self.df.iloc[idx]['label_idx']
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

class ImagePreprocessor:
    def __init__(self, time_steps=20, img_size=32):
        self.time_steps = time_steps
        self.img_size = img_size
        
    def preprocess_scan(self, image_tensor):
        device = image_tensor.device
        batch_size = image_tensor.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image_tensor[i].cpu().numpy().reshape(self.img_size, self.img_size)
            img_8bit = (img * 255).astype(np.uint8)
            
            blurred = cv2.GaussianBlur(img_8bit, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(img_8bit)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                skull_stripped = cv2.bitwise_and(img_8bit, mask)
            else:
                skull_stripped = img_8bit
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(skull_stripped)
            
            processed_img = torch.from_numpy(enhanced / 255.0).float().to(device)
            processed_images.append(processed_img.flatten())
        
        return torch.stack(processed_images, dim=0)
    
    def encode_temporal(self, data):
        device = data.device
        batch_size, features = data.shape
        
        data_norm = torch.clamp(data, 0, 1)
        spikes = torch.zeros(self.time_steps, batch_size, features, device=device)
        
        for b in range(batch_size):
            for f in range(features):
                intensity = data_norm[b, f].item()
                if intensity > 0.1:
                    spike_time = int((1 - intensity) * (self.time_steps - 1))
                    spike_time = max(0, min(spike_time, self.time_steps - 1))
                    spikes[spike_time, b, f] = 1.0
                    
                    if intensity > 0.8 and spike_time + 1 < self.time_steps:
                        spikes[spike_time + 1, b, f] = 0.5
        
        return spikes
    
    def extract_spike_features(self, spikes):
        features = []
        
        features.append(spikes.mean(dim=0))
        features.append(spikes.sum(dim=0))
        features.append(spikes.std(dim=0))
        features.append(spikes.max(dim=0)[0])
        
        batch_size, pixel_features = spikes.shape[1], spikes.shape[2]
        
        first_spike = torch.zeros(batch_size, pixel_features, device=spikes.device)
        for b in range(batch_size):
            for f in range(pixel_features):
                spike_times = torch.nonzero(spikes[:, b, f])
                if len(spike_times) > 0:
                    first_spike[b, f] = spike_times[0].float() / self.time_steps
                else:
                    first_spike[b, f] = 1.0
        features.append(first_spike)
        
        burst_features = torch.zeros_like(spikes[0])
        for t in range(1, self.time_steps):
            consecutive_spikes = spikes[t-1] * spikes[t]
            burst_features += consecutive_spikes
        features.append(burst_features)
        
        phase1 = spikes[:self.time_steps//2].mean(dim=0)
        phase2 = spikes[self.time_steps//2:].mean(dim=0)
        features.extend([phase1, phase2])
        
        return torch.cat(features, dim=1)
    
    def __call__(self, batch_data, add_noise=False):
        device = batch_data.device
        
        skull_stripped = self.preprocess_scan(batch_data)
        temporal_spikes = self.encode_temporal(skull_stripped)
        temporal_features = self.extract_spike_features(temporal_spikes)
        
        combined_features = temporal_features
        combined_features = torch.nan_to_num(combined_features, 0.0)
        combined_features = torch.clamp(combined_features, -10, 10)
        
        if add_noise:
            noise = torch.randn_like(combined_features) * 0.02
            noisy_features = combined_features + noise
            return noisy_features, combined_features
        
        return combined_features, combined_features

class FeatureAutoencoder(nn.Module):
    def __init__(self, input_size, time_steps, latent_dim=256):
        super(FeatureAutoencoder, self).__init__()
        self.input_size = input_size
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        
        self.encoder1 = nn.Linear(input_size, 512, bias=True)
        self.encoder2 = nn.Linear(512, 256, bias=True)
        self.bottleneck = nn.Linear(256, self.latent_dim, bias=True)
        self.decoder1 = nn.Linear(self.latent_dim, 256, bias=True)
        self.decoder2 = nn.Linear(256, 512, bias=True)
        self.decoder3 = nn.Linear(512, input_size, bias=True)
        
        self.threshold = 1.5
        self.ff_learning_rate = 0.01
        
        self.tau_pre = 20.0
        self.tau_post = 20.0
        self.A_plus = 0.02
        self.A_minus = 0.01
        
        self.reward_baseline = 0.0
        self.reward_lr = 0.2
        self.layer_norms = [0.3, 0.25, 0.2]
        
        self.init_weights()
    
    def init_weights(self):
        for i, layer in enumerate([self.encoder1, self.encoder2, self.bottleneck]):
            nn.init.normal_(layer.weight, mean=0.0, std=self.layer_norms[i])
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=0.1)
        for layer in [self.decoder1, self.decoder2, self.decoder3]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.2)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=0.1)
    
    def encode(self, input_features):
        with torch.no_grad():
            x1 = torch.relu(self.encoder1(input_features))
            x2 = torch.relu(self.encoder2(x1))
            z_m_raw = self.bottleneck(x2)
            z_m = torch.sigmoid(z_m_raw)
            return z_m.detach()
    
    def decode(self, z_m):
        with torch.no_grad():
            x = torch.relu(self.decoder1(z_m))
            x = torch.relu(self.decoder2(x))
            reconstruction = self.decoder3(x)
            return reconstruction
    
    def get_embeddings(self, dataloader, preprocessor, max_batches=None):
        embeddings = []
        labels_list = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                data = data.to(device)
                combined_features, _ = preprocessor(data)
                combined_features = combined_features.to(device)
                
                z_m = self.encode(combined_features)
                embeddings.append(z_m.cpu())
                labels_list.append(labels)
        
        return torch.cat(embeddings, dim=0), torch.cat(labels_list, dim=0)
    
    def forward(self, input_features):
        x1 = torch.relu(self.encoder1(input_features))
        x2 = torch.relu(self.encoder2(x1))
        z_m_raw = self.bottleneck(x2)
        z_m = torch.sigmoid(z_m_raw)
        
        d1 = torch.relu(self.decoder1(z_m))
        d2 = torch.relu(self.decoder2(d1))
        reconstruction = self.decoder3(d2)
        
        return reconstruction, z_m
    
    def create_samples(self, x, noise_level=0.2):
        batch_size = x.shape[0]
        positive_samples = x.clone()
        negative_samples = x.clone()
        
        noise = torch.randn_like(x) * noise_level * 2.0
        negative_samples += noise
        
        for i in range(batch_size):
            n_shuffle = int(0.4 * x.shape[1])
            shuffle_idx = torch.randperm(x.shape[1])[:n_shuffle]
            random_sample_idx = torch.randint(0, batch_size, (1,)).item()
            negative_samples[i, shuffle_idx] = x[random_sample_idx, shuffle_idx]
        
        invert_mask = torch.rand_like(x) < 0.2
        negative_samples[invert_mask] = 1.0 - negative_samples[invert_mask]
        
        random_mask = torch.rand_like(x) < 0.1
        negative_samples[random_mask] = torch.rand_like(negative_samples[random_mask])
        
        combined_samples = torch.cat([positive_samples, negative_samples], dim=0)
        labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)], dim=0)
        
        return combined_samples, labels
    
    def update_weights_direct(self, features):
        with torch.no_grad():
            x1 = torch.relu(self.encoder1(features))
            if x1.sum() < 1e-6:
                self.encoder1.weight.data += torch.randn_like(self.encoder1.weight) * 0.1
                self.encoder1.bias.data += torch.randn_like(self.encoder1.bias) * 0.1
            
            x2 = torch.relu(self.encoder2(x1))
            if x2.sum() < 1e-6:
                self.encoder2.weight.data += torch.randn_like(self.encoder2.weight) * 0.1
                self.encoder2.bias.data += torch.randn_like(self.encoder2.bias) * 0.1

            z_raw = self.bottleneck(x2)
            if z_raw.std() < 1e-6:
                self.bottleneck.weight.data += torch.randn_like(self.bottleneck.weight) * 0.15
                self.bottleneck.bias.data += torch.randn_like(self.bottleneck.bias) * 0.15
    
    def update_layer(self, layer, x_pos, x_neg, layer_idx):
        with torch.no_grad():
            h_pos = layer(x_pos)
            h_neg = layer(x_neg)
            
            if layer_idx < 2:
                h_pos_act = torch.relu(h_pos)
                h_neg_act = torch.relu(h_neg)
            else:
                h_pos_act = torch.sigmoid(h_pos)
                h_neg_act = torch.sigmoid(h_neg)
            
            goodness_pos = torch.sum(h_pos_act ** 2, dim=1)
            goodness_neg = torch.sum(h_neg_act ** 2, dim=1)
            
            pos_loss = torch.relu(self.threshold - goodness_pos) * 2.0
            neg_loss = torch.relu(goodness_neg - self.threshold) * 2.0

            total_grad = torch.zeros_like(layer.weight)
            bias_grad = torch.zeros_like(layer.bias)
            
            for i in range(len(x_pos)):
                if pos_loss[i] > 0:
                    grad = torch.outer(h_pos_act[i], x_pos[i]) * pos_loss[i].item()
                    total_grad += grad * 3.0
                    bias_grad += h_pos_act[i] * pos_loss[i].item() * 3.0
            
            for i in range(len(x_neg)):
                if neg_loss[i] > 0:
                    grad = torch.outer(h_neg_act[i], x_neg[i]) * neg_loss[i].item()
                    total_grad -= grad * 3.0
                    bias_grad -= h_neg_act[i] * neg_loss[i].item() * 3.0

            layer.weight.data += self.ff_learning_rate * (total_grad / (len(x_pos) + len(x_neg)))
            layer.bias.data += self.ff_learning_rate * (bias_grad / (len(x_pos) + len(x_neg)))
            
            layer.weight.data = torch.clamp(layer.weight.data, -0.5, 0.5)
            layer.bias.data = torch.clamp(layer.bias.data, -0.3, 0.3)
            
            return goodness_pos.mean(), goodness_neg.mean()
    
    def boost_weights(self):
        with torch.no_grad():
            self.encoder1.weight.data *= 1.5
            self.encoder1.bias.data += torch.randn_like(self.encoder1.bias) * 0.2
            
            self.encoder2.weight.data *= 1.5
            self.encoder2.bias.data += torch.randn_like(self.encoder2.bias) * 0.2
            
            self.bottleneck.weight.data *= 2.0
            self.bottleneck.bias.data += torch.randn_like(self.bottleneck.bias) * 0.3
    
    def train_model(self, data_loader, preprocessor, epochs=12):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            batch_count = 0
            for batch_idx, (data, labels) in enumerate(data_loader):
                if batch_idx >= FAST_TRAINING_BATCHES:
                    break
                
                data = data.to(device)
                features, _ = preprocessor(data)
                features = features.to(device)
                
                if batch_idx < 5:
                    self.update_weights_direct(features)
                
                combined_samples, sample_labels = self.create_samples(features)
                pos_samples = combined_samples[sample_labels == 1]
                neg_samples = combined_samples[sample_labels == 0]
                
                current_pos = pos_samples
                current_neg = neg_samples
                
                layers = [self.encoder1, self.encoder2, self.bottleneck]
                layer_goodness_pos = []
                layer_goodness_neg = []
                
                for layer_idx, layer in enumerate(layers):
                    goodness_pos, goodness_neg = self.update_layer(
                        layer, current_pos, current_neg, layer_idx
                    )
                    
                    layer_goodness_pos.append(goodness_pos)
                    layer_goodness_neg.append(goodness_neg)
                    
                    with torch.no_grad():
                        if layer_idx < len(layers) - 1:
                            current_pos = torch.relu(layer(current_pos))
                            current_neg = torch.relu(layer(current_neg))
                        else:
                            current_pos = torch.sigmoid(layer(current_pos))
                            current_neg = torch.sigmoid(layer(current_neg))
                
                with torch.no_grad():
                    test_z_m = self.encode(pos_samples[:4])
                    z_m_std = test_z_m.std().item()
                    
                    if z_m_std < 1e-6:
                        self.boost_weights()
                
                batch_count += 1
            
            if epoch % 3 == 0:
                print(f"  Processed {batch_count} batches")

def extract_features(autoencoder, dataloader, preprocessor, max_batches=None, dataset_type="unknown"):
    all_features = []
    all_labels = []
    
    desc = f"Extracting {dataset_type} features"
    if max_batches:
        desc += f" ({max_batches} batches)"
    else:
        desc += " (full dataset)"
    
    autoencoder.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc=desc)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            data = data.to(device)
            combined_features, _ = preprocessor(data)
            combined_features = combined_features.to(device)
            
            reconstruction, bottleneck_features = autoencoder(combined_features)
            
            final_features = torch.cat([combined_features, bottleneck_features], dim=1)
            
            all_features.append(final_features.cpu())
            all_labels.append(labels)
    
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    return features_tensor, labels_tensor

def select_features(train_features, train_labels, test_features, target_features=3000):
    print(f"Original features: {train_features.shape[1]}")
    X_train = train_features.numpy()
    X_test = test_features.numpy()
    y_train = train_labels.numpy()
    
    X_train = np.nan_to_num(X_train, 0.0)
    X_test = np.nan_to_num(X_test, 0.0)
    
    variance_selector = VarianceThreshold(threshold=0.005)
    X_train_var = variance_selector.fit_transform(X_train)
    X_test_var = variance_selector.transform(X_test)
    
    k_best = min(target_features * 3, X_train_var.shape[1])
    stat_selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_stat = stat_selector.fit_transform(X_train_var, y_train)
    X_test_stat = stat_selector.transform(X_test_var)
    
    rf_selector = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rfe_selector = RFE(estimator=rf_selector, n_features_to_select=target_features, step=300)
    X_train_final = rfe_selector.fit_transform(X_train_stat, y_train)
    X_test_final = rfe_selector.transform(X_test_stat)
    print(f"After RFE: {X_train_final.shape[1]}")
    
    return (X_train_final, X_test_final, 
            {'variance': variance_selector, 'statistical': stat_selector, 'rfe': rfe_selector})

def train_classifier(X_train, y_train, X_test, y_test):
    X_train = np.nan_to_num(X_train, 0.0)
    X_test = np.nan_to_num(X_test, 0.0)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_oob_score = rf.oob_score_
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.85,
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=12
    )
    gb.fit(X_train_scaled, y_train)
    
    ensemble = VotingClassifier([
        ('rf', rf),
        ('gb', gb)
    ], voting='soft')
    ensemble.fit(X_train_scaled, y_train)
    
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    rf_pred = rf.predict(X_test_scaled)
    gb_pred = gb.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    return ensemble, scaler, {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'precision_macro': precision_macro,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'oob_score': rf_oob_score,
        'rf_accuracy': rf_accuracy,
        'gb_accuracy': gb_accuracy
    }

def plot_results(y_true, y_pred, class_names, results, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    accuracy_per_class = []
    
    for i in range(len(class_names)):
        true_pos = cm[i, i]
        total_actual = cm[i, :].sum()
        accuracy_per_class.append(true_pos / total_actual if total_actual > 0 else 0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    total_samples = cm.sum()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'Confusion Matrix ({total_samples:,} samples)\nAccuracy: {results["accuracy"]:.3f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    x = np.arange(len(class_names))
    width = 0.2
    
    ax3.bar(x - width*1.5, precision_per_class, width, label='Precision', alpha=0.8, color='blue')
    ax3.bar(x - width*0.5, recall_per_class, width, label='Recall', alpha=0.8, color='orange')
    ax3.bar(x + width*0.5, f1_per_class, width, label='F1-Score', alpha=0.8, color='green')
    ax3.bar(x + width*1.5, accuracy_per_class, width, label='Class Accuracy', alpha=0.8, color='red')
    
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Class Performance Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    for i, (p, r, f, a) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, accuracy_per_class)):
        ax3.text(i-width*1.5, p+0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i-width*0.5, r+0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax3
