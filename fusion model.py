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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, 
                           precision_score, recall_score, accuracy_score)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import snntorch as snn
import cv2
from scipy import ndimage
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
TIME_STEPS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.008
EPOCHS = 10  # Reduced epochs
IMG_SIZE = 32
INPUT_SIZE = IMG_SIZE * IMG_SIZE

# TRAINING SPEED CONTROL - Only affects training, not evaluation
MAX_BATCHES_PER_EPOCH = 30  # Fast training: 30 batches * 64 samples = 1,920 samples per epoch
print(f"TRAINING OPTIMIZATION: Using {MAX_BATCHES_PER_EPOCH} batches per epoch for FAST training")
print(f"EVALUATION: Will use FULL dataset for accurate confusion matrix")

# Load Alzheimer's dataset
print("Loading Alzheimer's dataset...")
images = []
labels = []

def load_datasets(original_path, additional_path=None):
    """Load both original and additional datasets"""
    images = []
    labels = []
    
    # Load original dataset
    print("Loading original dataset...")
    for subfolder in tqdm(os.listdir(original_path)):
        subfolder_path = os.path.join(original_path, subfolder)
        for folder in os.listdir(subfolder_path):
            subfolder_path2 = os.path.join(subfolder_path, folder)
            for image_filename in os.listdir(subfolder_path2):
                if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(subfolder_path2, image_filename)
                    images.append(image_path)
                    labels.append(folder)
    
    # Load additional dataset
    if additional_path and os.path.exists(additional_path):
        print("Loading additional dataset...")
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
    if label == "NonDemented":
        return 0
    else:
        return 1

df['label_idx'] = df['label'].apply(create_binary_labels)
label_to_idx = {'NonDemented' : 0, 'Demented' : 1}

print("Dataset loaded successfully!")
print(f"Classes: {label_to_idx}")
print(f"Total samples: {len(df)}")
print(f"Class distribution: {df['label'].value_counts()}")

class AlzheimerDataset(Dataset):
    """Dataset class for Alzheimer's brain scans"""
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

class NeuromorphicSingleEncodingPreprocessor:
    """Optimized neuromorphic preprocessor with efficient Temporal encoding only"""
    
    def __init__(self, time_steps=20, img_size=32):
        self.time_steps = time_steps
        self.img_size = img_size
        print(f"Neuromorphic Optimized Temporal Encoding Preprocessor with {time_steps} time steps")
        
    def skull_strip_brain_scan(self, image_tensor):
        """Advanced skull stripping for brain scans"""
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
    
    def neuromorphic_temporal_encoding(self, data):
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
    
    def extract_neuromorphic_features(self, spikes):
        """Extract essential temporal features only"""
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
        """Main neuromorphic preprocessing pipeline"""
        device = batch_data.device
        
        skull_stripped = self.skull_strip_brain_scan(batch_data)
        temporal_spikes = self.neuromorphic_temporal_encoding(skull_stripped)
        temporal_features = self.extract_neuromorphic_features(temporal_spikes)
        
        combined_features = temporal_features
        combined_features = torch.nan_to_num(combined_features, 0.0)
        combined_features = torch.clamp(combined_features, -10, 10)
        
        if add_noise:
            noise = torch.randn_like(combined_features) * 0.02
            noisy_features = combined_features + noise
            return noisy_features, combined_features
        
        return combined_features, combined_features

class ForwardForwardRSTDPAutoencoder(nn.Module):
    
    def __init__(self, input_size, time_steps, latent_dim=256):
        super(ForwardForwardRSTDPAutoencoder, self).__init__()
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
        
        self.init_weights_aggressive()
        
        print(f"  Architecture: {input_size} -> 512 -> 256 -> {self.latent_dim}(z_m) -> 256 -> 512 -> {input_size}")
        print(f"  FF Learning Rate: {self.ff_learning_rate}")
        print(f"  Layer Norms: {self.layer_norms}")
    
    def init_weights_aggressive(self):
        for i, layer in enumerate([self.encoder1, self.encoder2, self.bottleneck]):
            nn.init.normal_(layer.weight, mean=0.0, std=self.layer_norms[i])
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=0.1)
        for layer in [self.decoder1, self.decoder2, self.decoder3]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.2)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=0.1)
    
    def encode_to_z_m(self, input_features):
        with torch.no_grad():
            x1 = torch.relu(self.encoder1(input_features))
            x2 = torch.relu(self.encoder2(x1))
            z_m_raw = self.bottleneck(x2)
            z_m = torch.sigmoid(z_m_raw)
            return z_m.detach()
    
    def decode_from_z_m(self, z_m):
        with torch.no_grad():
            x = torch.relu(self.decoder1(z_m))
            x = torch.relu(self.decoder2(x))
            reconstruction = self.decoder3(x)
            return reconstruction
    
    def get_z_m_batch(self, dataloader, preprocessor, max_batches=None):
        """Extract z_m from dataloader - NO LIMIT for evaluation"""
        z_m_vectors = []
        labels_list = []
        
        self.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Extracting z_m from FULL dataset")):
                # EVALUATION: No batch limit for full dataset processing
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                data = data.to(device)
                combined_features, _ = preprocessor(data)
                combined_features = combined_features.to(device)
                
                z_m = self.encode_to_z_m(combined_features)
                z_m_vectors.append(z_m.cpu())
                labels_list.append(labels)
        
        return torch.cat(z_m_vectors, dim=0), torch.cat(labels_list, dim=0)
    
    def forward(self, input_features):
        x1 = torch.relu(self.encoder1(input_features))
        x2 = torch.relu(self.encoder2(x1))
        z_m_raw = self.bottleneck(x2)
        z_m = torch.sigmoid(z_m_raw)
        
        d1 = torch.relu(self.decoder1(z_m))
        d2 = torch.relu(self.decoder2(d1))
        reconstruction = self.decoder3(d2)
        
        return reconstruction, z_m
    
    def create_positive_negative_samples(self, x, noise_level=0.2):
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
    
    def direct_weight_updates(self, features):
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
    
    def aggressive_ff_update(self, layer, x_pos, x_neg, layer_idx):
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
    
    def emergency_weight_boost(self):
        with torch.no_grad():
            self.encoder1.weight.data *= 1.5
            self.encoder1.bias.data += torch.randn_like(self.encoder1.bias) * 0.2
            
            self.encoder2.weight.data *= 1.5
            self.encoder2.bias.data += torch.randn_like(self.encoder2.bias) * 0.2
            
            self.bottleneck.weight.data *= 2.0
            self.bottleneck.bias.data += torch.randn_like(self.bottleneck.bias) * 0.3
    
    def train_ff_rstdp_neuromorphic(self, data_loader, preprocessor, epochs=6, max_batches_per_epoch=None):
        """FAST TRAINING: Limited batches per epoch for speed"""
        
        epoch_pos_goodness = []
        epoch_neg_goodness = []
        epoch_goodness_gap = []
        epoch_recon_loss = []
        epoch_z_m_means = []
        epoch_z_m_stds = []
        epoch_z_m_separability = []
        epoch_reward = []
        
        print("\n" + "="*80)
        print("FAST SNN TRAINING WITH LIMITED BATCHES PER EPOCH")
        print(f"Training batches per epoch: {max_batches_per_epoch}")
        print("="*80)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs} - Fast Training Mode")
            print("-" * 60)
            
            batch_pos_goodness = []
            batch_neg_goodness = []
            batch_recon_losses = []
            batch_z_m_means = []
            batch_z_m_stds = []
            batch_z_m_separability = []
            batch_rewards = []
            
            # FAST TRAINING: Process only limited batches per epoch
            for batch_idx, (data, labels) in enumerate(tqdm(data_loader, desc=f"Fast Training Epoch {epoch+1}")):
                # TRAINING SPEED LIMIT: Stop after max_batches_per_epoch
                if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                    break
                
                data = data.to(device)
                features, _ = preprocessor(data)
                features = features.to(device)
                
                if batch_idx < 3:
                    self.direct_weight_updates(features)
                
                combined_samples, sample_labels = self.create_positive_negative_samples(features)
                pos_samples = combined_samples[sample_labels == 1]
                neg_samples = combined_samples[sample_labels == 0]
                
                current_pos = pos_samples
                current_neg = neg_samples
                
                layers = [self.encoder1, self.encoder2, self.bottleneck]
                layer_goodness_pos = []
                layer_goodness_neg = []
                
                for layer_idx, layer in enumerate(layers):
                    goodness_pos, goodness_neg = self.aggressive_ff_update(
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
                
                avg_pos_goodness = torch.stack(layer_goodness_pos).mean().item()
                avg_neg_goodness = torch.stack(layer_goodness_neg).mean().item()
                
                batch_pos_goodness.append(avg_pos_goodness)
                batch_neg_goodness.append(avg_neg_goodness)
                
                with torch.no_grad():
                    original_data_z_m = self.encode_to_z_m(features[:min(8, len(features))])
                    original_labels = labels[:min(8, len(labels))]
                    
                    z_m_mean = original_data_z_m.mean().item()
                    z_m_std = original_data_z_m.std().item()
                    
                    if len(torch.unique(original_labels)) > 1:
                        class_0_z_m = original_data_z_m[original_labels == 0]
                        class_1_z_m = original_data_z_m[original_labels == 1]
                        
                        if len(class_0_z_m) > 0 and len(class_1_z_m) > 0:
                            mean_0 = class_0_z_m.mean(dim=0)
                            mean_1 = class_1_z_m.mean(dim=0)
                            separability = torch.norm(mean_0 - mean_1).item()
                        else:
                            separability = 0.0
                    else:
                        separability = 0.0
                    
                    batch_z_m_means.append(z_m_mean)
                    batch_z_m_stds.append(z_m_std)
                    batch_z_m_separability.append(separability)
                    
                    if z_m_std < 1e-6:
                        self.emergency_weight_boost()
                        print(f"    EMERGENCY: z_m variance collapsed! Applied weight boost.")
                
                with torch.no_grad():
                    reconstruction, _ = self.forward(pos_samples[:4])
                    recon_loss = F.mse_loss(reconstruction, pos_samples[:4]).item()
                    reward_signal = -recon_loss
                    
                    batch_recon_losses.append(recon_loss)
                    batch_rewards.append(reward_signal)
                
                # Fast training: Print every 10 batches instead of 50
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx+1}:")
                    print(f"    FF: Pos={avg_pos_goodness:.3f}, Neg={avg_neg_goodness:.3f}, Gap={avg_pos_goodness-avg_neg_goodness:.3f}")
                    print(f"    z_m: Mean={z_m_mean:.6f}, Std={z_m_std:.6f}, Sep={separability:.4f}")
            
            # Aggregate epoch statistics
            epoch_pos = np.mean(batch_pos_goodness) if batch_pos_goodness else 0
            epoch_neg = np.mean(batch_neg_goodness) if batch_neg_goodness else 0
            epoch_gap = epoch_pos - epoch_neg
            epoch_recon = np.mean(batch_recon_losses) if batch_recon_losses else 0
            epoch_z_m_mean = np.mean(batch_z_m_means) if batch_z_m_means else 0
            epoch_z_m_std = np.mean(batch_z_m_stds) if batch_z_m_stds else 0
            epoch_separability = np.mean(batch_z_m_separability) if batch_z_m_separability else 0
            epoch_rew = np.mean(batch_rewards) if batch_rewards else 0
            
            epoch_pos_goodness.append(epoch_pos)
            epoch_neg_goodness.append(epoch_neg)
            epoch_goodness_gap.append(epoch_gap)
            epoch_recon_loss.append(epoch_recon)
            epoch_z_m_means.append(epoch_z_m_mean)
            epoch_z_m_stds.append(epoch_z_m_std)
            epoch_z_m_separability.append(epoch_separability)
            epoch_reward.append(epoch_rew)
            
            print(f"\nFast Epoch {epoch+1} Summary:")
            print(f"  FF Training: Pos={epoch_pos:.4f}, Neg={epoch_neg:.4f}, Gap={epoch_gap:.4f}")
            print(f"  z_m Quality: Mean={epoch_z_m_mean:.6f}, Std={epoch_z_m_std:.6f}, Sep={epoch_separability:.4f}")
            print(f"  Reconstruction: Loss={epoch_recon:.6f}, Reward={epoch_rew:.6f}")
            print(f"  Processed {len(batch_pos_goodness)} batches (limited for speed)")
            
            if epoch > 0:
                z_m_improvement = epoch_z_m_std - epoch_z_m_stds[epoch-1] if len(epoch_z_m_stds) > 1 else 0
                separability_improvement = epoch_separability - epoch_z_m_separability[epoch-1] if len(epoch_z_m_separability) > 1 else 0
                print(f"  Progress: z_m_std {z_m_improvement:+.6f}, Sep {separability_improvement:+.4f}")
        
        print(f"\n{'='*80}")
        print("FAST SNN TRAINING COMPLETED")
        print("="*80)
        
        return {
            'pos_goodness': epoch_pos_goodness,
            'neg_goodness': epoch_neg_goodness,
            'goodness_gap': epoch_goodness_gap,
            'reconstruction_loss': epoch_recon_loss,
            'z_m_means': epoch_z_m_means,
            'z_m_stds': epoch_z_m_stds,
            'z_m_separability': epoch_z_m_separability,
            'rewards': epoch_reward
        }

def extract_z_m_latent_variables_aggressive(autoencoder, train_loader, test_loader, preprocessor, save_path=None):
    """FULL DATASET EVALUATION: Extract z_m from complete dataset"""
    print(f"\nEVALUATION MODE: Extracting z_m from FULL dataset...")
    
    # Test single sample first
    single_batch = next(iter(train_loader))
    single_data = single_batch[0][:1].to(device)
    single_features, _ = preprocessor(single_data)
    single_features = single_features.to(device)
    
    with torch.no_grad():
        test_z_m = autoencoder.encode_to_z_m(single_features)
        print(f"Single sample z_m:")
        print(f"  Shape: {test_z_m.shape}")
        print(f"  Mean: {test_z_m.mean().item():.8f}")
        print(f"  Std: {test_z_m.std().item():.8f}")
    
    if test_z_m.std().item() < 1e-7:
        with torch.no_grad():
            nn.init.normal_(autoencoder.bottleneck.weight, mean=0.0, std=0.5)
            nn.init.normal_(autoencoder.bottleneck.bias, mean=0.2, std=0.1)
        
        with torch.no_grad():
            test_z_m = autoencoder.encode_to_z_m(single_features)
            print(f"After emergency fix:")
            print(f"  Mean: {test_z_m.mean().item():.8f}")
            print(f"  Std: {test_z_m.std().item():.8f}")
        
        if test_z_m.std().item() < 1e-7:
            with torch.no_grad():
                x1 = torch.relu(autoencoder.encoder1(single_features))
                z_m_bypass = torch.sigmoid(autoencoder.encoder2(x1))
            print(f"Using encoder2 as z_m bypass:")
            print(f"  Mean: {z_m_bypass.mean().item():.8f}")
            print(f"  Std: {z_m_bypass.std().item():.8f}")
            
            def encode_to_z_m_bypass(self, input_features):
                with torch.no_grad():
                    x1 = torch.relu(self.encoder1(input_features))
                    z_m = torch.sigmoid(self.encoder2(x1))
                    return z_m.detach()
            
            autoencoder.encode_to_z_m = encode_to_z_m_bypass.__get__(autoencoder, ForwardForwardRSTDPAutoencoder)
    
    # FULL DATASET EXTRACTION: No batch limits for evaluation
    print(f"Extracting z_m from COMPLETE training set (NO LIMITS)...")
    train_z_m, train_labels = autoencoder.get_z_m_batch(train_loader, preprocessor)  # No max_batches
    
    print(f"Extracting z_m from COMPLETE test set (NO LIMITS)...")
    test_z_m_full, test_labels = autoencoder.get_z_m_batch(test_loader, preprocessor)  # No max_batches
    
    # Final statistics
    print(f"\nFULL DATASET z_m EXTRACTION RESULTS:")
    print(f"  Train z_m shape: {train_z_m.shape}")
    print(f"  Test z_m shape: {test_z_m_full.shape}")
    print(f"  z_m dimension: {train_z_m.shape[1]}")
    
    train_mean = train_z_m.mean().item()
    train_std = train_z_m.std().item()
    train_min = train_z_m.min().item()
    train_max = train_z_m.max().item()
    
    print(f"\nFull Dataset z_m Statistics:")
    print(f"  Mean: {train_mean:.8f}")
    print(f"  Std:  {train_std:.8f}")
    print(f"  Min:  {train_min:.8f}")
    print(f"  Max:  {train_max:.8f}")
    
    sparsity = (train_z_m == 0).float().mean().item()
    print(f"  Sparsity: {sparsity:.4%}")
    
    if train_std > 1e-6:
        print(f"SUCCESS: Non-zero variance: {train_std:.8f}")
        print(f"Range: [{train_min:.4f}, {train_max:.4f}]")
        success = True
    else:
        success = False
    
    z_m_data = {
        'train_z_m': train_z_m,
        'train_labels': train_labels,
        'test_z_m': test_z_m_full,
        'test_labels': test_labels,
        'latent_dim': train_z_m.shape[1],
        'modality': 'MRI',
        'success': success,
        'method': 'full_dataset_evaluation'
    }
    
    if save_path:
        torch.save(z_m_data, save_path)
    
    return z_m_data

def advanced_feature_selection(train_features, train_labels, test_features, target_features=3000):
    print(f"\nAdvanced Feature Selection")
    print(f"Original features: {train_features.shape[1]}")
    
    X_train = train_features.numpy()
    X_test = test_features.numpy()
    y_train = train_labels.numpy()
    
    X_train = np.nan_to_num(X_train, 0.0)
    X_test = np.nan_to_num(X_test, 0.0)
    
    # Variance filtering
    variance_selector = VarianceThreshold(threshold=0.01)
    X_train_var = variance_selector.fit_transform(X_train)
    X_test_var = variance_selector.transform(X_test)
    print(f"After variance filtering: {X_train_var.shape[1]}")
    
    # Statistical selection
    k_best = min(target_features * 2, X_train_var.shape[1])
    stat_selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_stat = stat_selector.fit_transform(X_train_var, y_train)
    X_test_stat = stat_selector.transform(X_test_var)
    print(f"After statistical selection: {X_train_stat.shape[1]}")
    
    # RFE selection
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe_selector = RFE(estimator=rf_selector, n_features_to_select=target_features, step=500)
    X_train_final = rfe_selector.fit_transform(X_train_stat, y_train)
    X_test_final = rfe_selector.transform(X_test_stat)
    print(f"After RFE: {X_train_final.shape[1]}")
    
    return (X_train_final, X_test_final, 
            {'variance': variance_selector, 'statistical': stat_selector, 'rfe': rfe_selector})

def train_fast_ensemble_classifier(X_train, y_train, X_test, y_test):
    print(f"\nTraining fast ensemble classifier")
    
    X_train = np.nan_to_num(X_train, 0.0)
    X_test = np.nan_to_num(X_test, 0.0)
    
    # Preprocessing
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        oob_score=True,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_oob_score = rf.oob_score_
    
    print("Training Gradient Boosting")
    gb = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.15,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=8
    )
    gb.fit(X_train_scaled, y_train)
    
    print("Creating ensemble")
    ensemble = VotingClassifier([
        ('rf', rf),
        ('gb', gb)
    ], voting='soft')
    ensemble.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = ensemble.predict(X_test_scaled)
    y_pred_proba = ensemble.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Individual performance
    rf_pred = rf.predict(X_test_scaled)
    gb_pred = gb.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    
    print(f"Random Forest accuracy: {rf_accuracy:.4f}")
    print(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
    print(f"Ensemble accuracy: {accuracy:.4f}")
    print(f"Random Forest OOB Score: {rf_oob_score:.4f}")
    
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

def generate_comprehensive_confusion_matrix(y_true, y_pred, class_names, results, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    accuracy_per_class = []
    
    for i in range(len(class_names)):
        true_pos = cm[i, i]
        total_actual = cm[i, :].sum()
        accuracy_per_class.append(true_pos / total_actual if total_actual > 0 else 0)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Raw confusion matrix - SHOWS LARGE COUNTS FROM FULL DATASET
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'Confusion Matrix (FULL DATASET - Large Raw Counts)\nAccuracy: {results["accuracy"]:.3f}')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # Per-class metrics bar chart
    x = np.arange(len(class_names))
    width = 0.2
    
    ax3.bar(x - width*1.5, precision_per_class, width, label='Precision', alpha=0.8, color='blue')
    ax3.bar(x - width*0.5, recall_per_class, width, label='Recall', alpha=0.8, color='orange')
    ax3.bar(x + width*0.5, f1_per_class, width, label='F1-Score', alpha=0.8, color='green')
    ax3.bar(x + width*1.5, accuracy_per_class, width, label='Class Accuracy', alpha=0.8, color='red')
    
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Class Performance Metrics (FULL DATASET)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (p, r, f, a) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, accuracy_per_class)):
        ax3.text(i-width*1.5, p+0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i-width*0.5, r+0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i+width*0.5, f+0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i+width*1.5, a+0.01, f'{a:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Performance summary - EMPHASIZES FULL DATASET PROCESSING
    total_samples = cm.sum()
    summary_text = f"""
FAST TRAINING + FULL DATASET EVALUATION
Neuromorphic System Performance Summary


Total Test Samples:   {total_samples:,}
Overall Accuracy:     {results['accuracy']:.4f}
Precision (weighted): {results['precision_weighted']:.4f}
Precision (macro):    {results['precision_macro']:.4f}
Recall (weighted):    {results['recall_weighted']:.4f}
F1-Score (weighted):  {results['f1_weighted']:.4f}
OOB Score:            {results['oob_score']:.4f}

Individual Classifiers:
Random Forest:      {results['rf_accuracy']:.4f}
Gradient Boosting:  {results['gb_accuracy']:.4f}
Ensemble:           {results['accuracy']:.4f}

Per-Class Results:
NonDemented:
  Precision:   {precision_per_class[0]:.3f}
  Recall:      {recall_per_class[0]:.3f}
  F1-Score:    {f1_per_class[0]:.3f}
  
Demented:
  Precision:   {precision_per_class[1]:.3f}
  Recall:      {recall_per_class[1]:.3f}
  F1-Score:    {f1_per_class[1]:.3f}

Raw Confusion Matrix (LARGE COUNTS):
NonDemented: {cm[0,0]:,} correct, {cm[0,1]:,} misclassified
Demented: {cm[1,1]:,} correct, {cm[1,0]:,} misclassified

Training: Limited batches for SPEED
Evaluation: Full dataset for ACCURACY
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax4.set_title('Fast Training + Full Dataset Evaluation Results')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'confusion_matrix': cm,
        'normalized_cm': cm_normalized,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'accuracy_per_class': accuracy_per_class,
        'results': results
    }

def save_neuromorphic_model(autoencoder, ensemble, scaler, selectors, preprocessor, 
                           save_dir='fast_train_full_eval_model'):
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'architecture': autoencoder
    }, os.path.join(save_dir, 'fast_train_autoencoder.pth'))
    
    model_components = {
        'ensemble': ensemble,
        'scaler': scaler,
        'selectors': selectors,
        'preprocessor': preprocessor
    }
    
    with open(os.path.join(save_dir, 'fast_train_components.pkl'), 'wb') as f:
        pickle.dump(model_components, f)

def extract_neuromorphic_features(autoencoder, dataloader, preprocessor):
    """FULL DATASET FEATURE EXTRACTION: Process ALL batches for evaluation"""
    all_features = []
    all_labels = []
    
    autoencoder.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Extracting features from FULL dataset")):
            # EVALUATION MODE: NO batch limits - process all data
            
            data = data.to(device)
            combined_features, _ = preprocessor(data)
            combined_features = combined_features.to(device)
            
            reconstruction, bottleneck_features = autoencoder(combined_features)
            
            final_features = torch.cat([combined_features, bottleneck_features], dim=1)
            
            all_features.append(final_features.cpu())
            all_labels.append(labels)
    
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print(f"FULL DATASET feature extraction complete:")
    print(f"  Features shape: {features_tensor.shape}")
    print(f"  Labels shape: {labels_tensor.shape}")
    
    return features_tensor, labels_tensor

# Data preparation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
])

# Split data
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['label_idx'], random_state=42
)

ADDITIONAL_TEST_PATH = '/kaggle/input/nins-dementia-v2'

def load_additional_test_data(additional_path):
    """Load additional test data from Demented/NonDemented folders"""
    additional_images = []
    additional_labels = []
    
    if additional_path and os.path.exists(additional_path):
        print("Loading additional test data...")
        for class_folder in ['Demented', 'NonDemented']:
            class_path = os.path.join(additional_path, class_folder)
            if os.path.exists(class_path):
                print(f"Processing {class_folder} folder...")
                for root, dirs, files in os.walk(class_path):
                    for file in tqdm(files, desc=f"Loading {class_folder}"):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            image_path = os.path.join(root, file)
                            additional_images.append(image_path)
                            additional_labels.append(class_folder)
        
        print(f"Loaded {len(additional_images)} additional test images")
        return additional_images, additional_labels
    else:
        print(f"Additional dataset path not found: {additional_path}")
        return [], []

# Load and add additional test data
additional_images, additional_labels = load_additional_test_data(ADDITIONAL_TEST_PATH)

if additional_images:
    additional_df = pd.DataFrame({
        'image': additional_images, 
        'label': additional_labels
    })
    
    additional_df['label_idx'] = additional_df['label'].apply(create_binary_labels)
    
    original_test_size = len(test_df)
    test_df = pd.concat([test_df, additional_df], ignore_index=True)
    
    print(f"Original test set size: {original_test_size}")
    print(f"Additional test data: {len(additional_df)}")
    print(f"Combined test set size: {len(test_df)}")
    print(f"Test set class distribution:\n{test_df['label'].value_counts()}")
else:
    print("No additional test data loaded")

print(f"  Total train samples: {len(train_df)}")
print(f"  Total test samples: {len(test_df)}")

train_subset = train_df
test_subset = test_df

# Create datasets
train_dataset = AlzheimerDataset(train_subset, transform=transform)
test_dataset = AlzheimerDataset(test_subset, transform=transform)

# Create weighted sampler
train_labels_list = train_subset['label_idx'].tolist()
class_counts = Counter(train_labels_list)
class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in train_labels_list]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels_list))

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"Classes: {label_to_idx}")
print(f"Train distribution: {Counter(train_labels_list)}")
print(f"Test distribution: {Counter(test_subset['label_idx'].tolist())}")

# MAIN EXECUTION: FAST TRAINING + FULL EVALUATION
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("HYBRID APPROACH: FAST TRAINING + FULL DATASET EVALUATION")
    print("="*80)
    print(f"Training: LIMITED to {MAX_BATCHES_PER_EPOCH} batches per epoch for SPEED")
    print(f"Evaluation: FULL dataset processing for LARGE confusion matrix counts")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = NeuromorphicSingleEncodingPreprocessor(TIME_STEPS, img_size=IMG_SIZE)

    sample_batch = next(iter(train_loader))[0][:4].to(device)
    sample_features, _ = preprocessor(sample_batch)
    combined_dim = sample_features.shape[1]
   
    autoencoder = ForwardForwardRSTDPAutoencoder(combined_dim, TIME_STEPS).to(device)

    print("\nPHASE 1: SNN TRAINING")
    print("-" * 50)
    training_metrics = autoencoder.train_ff_rstdp_neuromorphic(
        train_loader, preprocessor, epochs=EPOCHS, 
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH  # SPEED OPTIMIZATION
    )
    
    print("\nPHASE 2: FULL DATASET EVALUATION")
    print("-" * 50)
    z_m_data = extract_z_m_latent_variables_aggressive(
        autoencoder, train_loader, test_loader, preprocessor, 
        save_path='fast_train_full_eval_z_m.pth'
    )
    
    if z_m_data['success']:
        print(f"z_m extraction successful!")
        print(f"z_m mean: {z_m_data['train_z_m'].mean().item():.6f}")
        print(f"z_m std:  {z_m_data['train_z_m'].std().item():.6f}")
        
        # PHASE 3: FULL DATASET FEATURE EXTRACTION
        print("\nPHASE 3: FULL DATASET FEATURE EXTRACTION")
        print("-" * 50)
        train_features, train_labels = extract_neuromorphic_features(autoencoder, train_loader, preprocessor)
        test_features, test_labels = extract_neuromorphic_features(autoencoder, test_loader, preprocessor)
        
        print(f"FULL DATASET RESULTS:")
        print(f"  Train features: {train_features.shape}")  # LARGE numbers
        print(f"  Test features: {test_features.shape}")    # LARGE numbers
        
        # Advanced feature selection
        X_train_selected, X_test_selected, selectors = advanced_feature_selection(
            train_features, train_labels, test_features, target_features=3000
        )
        
        # Train fast ensemble
        ensemble, scaler, results = train_fast_ensemble_classifier(
            X_train_selected, train_labels.numpy(), X_test_selected, test_labels.numpy()
        )
        
        # Generate comprehensive results with LARGE confusion matrix counts
        class_names = ['NonDemented', 'Demented']
        
        comprehensive_results = generate_comprehensive_confusion_matrix(
            test_labels.numpy(), results['predictions'], class_names, results
        )
        
        # Save complete neuromorphic model
        save_neuromorphic_model(autoencoder, ensemble, scaler, selectors, preprocessor)
        
        print(f"\n" + "="*80)
        print(f"Evaluation: ({test_labels.shape[0]:,} test samples)")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision (weighted): {results['precision_weighted']:.4f}")
        print(f"  F1-Score (weighted): {results['f1_weighted']:.4f}")
        print(f"  OOB Score: {results['oob_score']:.4f}")
        
        # Display actual confusion matrix counts
        cm = comprehensive_results['confusion_matrix']
        print(f"\nCONFUSION MATRIX RAW COUNTS:")
        print(f"  NonDemented: {cm[0,0]:,} correct, {cm[0,1]:,} misclassified")
        print(f"  Demented: {cm[1,1]:,} correct, {cm[1,0]:,} misclassified")
        print(f"  Total test samples: {cm.sum():,}")
        print("="*80)
    
    else:
        print("z_m extraction failed. Check model training.")
