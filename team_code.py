#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################
import os
import numpy as np
import torch
import torch.nn as nn
import wfdb
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer import ECGTransformer
from tqdm import tqdm
from scipy.signal import savgol_filter, medfilt
from helper_code import *
from torch.cuda.amp import autocast, GradScaler
from imblearn.combine import SMOTETomek
from collections import Counter
from joblib import Parallel, delayed

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def get_median_filter_width(sampling_rate, duration):
    res = int(sampling_rate * duration)
    res += ((res % 2) - 1)  # Ensure odd filter width
    return res

def filter_signal_with_baseline(X, sampling_rate, ms_flt_array):
    X_filtered = np.zeros_like(X)
    for lead in range(X.shape[1]):
        X0 = X[:, lead]
        for duration in ms_flt_array:
            filter_width = get_median_filter_width(sampling_rate, duration)
            X0 = medfilt(X0, kernel_size=filter_width)
        X_filtered[:, lead] = X[:, lead] - X0
    return X_filtered

def load_ecg_signal(record_path, target_length=1024, target_leads=12):
    try:
        signal, metadata = wfdb.rdsamp(record_path)
        sampling_rate = metadata['fs']
        num_leads = signal.shape[1]
    except Exception as e:
        print(f"Error loading record {record_path}: {e}")
        return np.zeros((target_length, target_leads), dtype=np.float16)

    # Adjust leads if necessary
    if num_leads != target_leads:
        print(f"⚠️ Warning: Expected {target_leads} leads, got {num_leads} in {record_path}. Adjusting...")
        signal = signal[:, :target_leads] if num_leads > target_leads else np.pad(signal, ((0, 0), (0, target_leads - num_leads)), mode='constant')

    # Adjust length
    current_length = signal.shape[0]
    if current_length < target_length:
        pad_width = target_length - current_length
        signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')
    else:
        signal = signal[:target_length, :]

    # Vectorized Savitzky-Golay filter (faster)
    smoothed_signal = np.array([savgol_filter(signal[:, lead], 11, 3) for lead in range(target_leads)]).T

    # Apply baseline filter
    ms_flt_array = [0.2, 0.6]
    return filter_signal_with_baseline(smoothed_signal, sampling_rate, ms_flt_array).astype(np.float16)

def load_label(record_path):
    try:
        hea_file = f"{record_path}.hea"
        with open(hea_file, 'r') as f:
            for line in f:
                if "Chagas label:" in line:
                    return 1 if "True" in line else 0
    except Exception as e:
        print(f"Error reading label from {record_path}: {e}")
    return None  # Return None for invalid labels

class ECGDataset(Dataset):
    def __init__(self, data_folder, records, target_length=1024, target_leads=12):
        self.data_folder = data_folder
        self.records = records
        self.target_length = target_length
        self.target_leads = target_leads

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record_name = self.records[idx]
        record_path = os.path.join(self.data_folder, record_name)
        
        for _ in range(3):  # Retry 3 times
            signal = load_ecg_signal(record_path, self.target_length, self.target_leads)
            if signal is not None:
                label = load_label(record_path)
                return torch.tensor(signal.T, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

        print(f"⚠️ Skipping {record_path} due to repeated failures.")
        return torch.zeros((12, self.target_length)), torch.tensor(0, dtype=torch.long)  # Return valid default


def check_class_imbalance(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]  # Convert tensor to int
    label_counts = Counter(labels)
    
    print("\n📊 Class Distribution in Dataset:")
    for label, count in label_counts.items():
        print(f"Class {label}: {count} samples")

    if len(label_counts) < 2:
        raise ValueError("Dataset contains only one class. Cannot proceed with training.")

    return label_counts


def extract_features(record_name, data_folder, target_length=1024, target_leads=12):
    record_path = os.path.join(data_folder, record_name)
    signal = load_ecg_signal(record_path, target_length, target_leads)
    label = load_label(record_path)
    if label is not None:
        return signal.flatten(), label
    return None

def apply_smotetomek_parallel(data_folder, records, target_length=1024, target_leads=12, n_jobs=-1):
    print("⚙️ Extracting features in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_features)(record, data_folder, target_length, target_leads)
        for record in tqdm(records, desc="Parallel Feature Extraction")
    )
    results = [res for res in results if res is not None]

    features, labels = zip(*results)
    features = np.array(features)
    labels = np.array(labels)

    print(f"\n📊 Original class distribution: {Counter(labels)}")


    print("⚖️ Applying SMOTE-Tomek...")
    smote_tomek = SMOTETomek(n_jobs=n_jobs)
    X_resampled, y_resampled = smote_tomek.fit_resample(features, labels)

    print(f"✅ Resampled class distribution: {Counter(y_resampled)}")

    X_resampled = X_resampled.reshape(-1, target_leads, target_length)
    return X_resampled, y_resampled

def save_checkpoint(model, optimizer, epoch, model_folder, best=False):
    os.makedirs(model_folder, exist_ok=True)
    checkpoint_path = os.path.join(model_folder, f"checkpoints/Tepoch_{epoch}.pth")
    best_model_path = os.path.join(model_folder, "Tbest_model.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if best:
        torch.save(state, best_model_path)
        print(f"🌟 Best model saved at: {best_model_path}")
    else:
        torch.save(state, checkpoint_path)
        print(f"✅ Checkpoint saved at: {checkpoint_path}")
        
def train_model(data_folder, model_folder, verbose=True):
    records = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.hea')]
    if not records:
        raise FileNotFoundError("No ECG data found in the folder.")
    
    dataset = ECGDataset(data_folder, records)
    # Display class imbalance before applying SMOTE-Tomek
    check_class_imbalance(dataset)

     # Apply SMOTE-Tomek
    #X_resampled, y_resampled = apply_smotetomek(dataset)
    #dataset.data, dataset.labels = X_resampled, y_resampled  # Replace dataset with balanced data
    
    # Apply SMOTE-Tomek
    X_resampled, y_resampled = apply_smotetomek_parallel(data_folder, records)
    dataset.data, dataset.labels = X_resampled, y_resampled
    
    # Display class distribution after balancing
    print("\n✅ After SMOTE-Tomek:")
    label_counts_after = Counter(y_resampled)
    for label, count in label_counts_after.items():
        print(f"Class {label}: {count} samples")
        
    # Step 5: Manual stratified split
    #indices_per_class = {label: [] for label in label_counts.keys()}
    #for idx, label in enumerate(y_resampled):
    #    indices_per_class[label].append(idx)

    indices_per_class = {label: [] for label in label_counts_after.keys()}
    for idx, label in enumerate(y_resampled):
        indices_per_class[label].append(idx)


    train_indices, val_indices = [], []

     # Ensure reproducibility
    #random.seed(42)
    torch.random.manual_seed(42)

    for label, indices in indices_per_class.items():
        # Shuffle indices to randomize within class
        #torch.random.manual_seed(42)  # for reproducibility
        indices = torch.tensor(indices)
        indices = indices[torch.randperm(len(indices))].tolist()

        split = int(0.8 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    # Final datasets # Step 6: Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

     # Step 6: Check balance inside train and val splits
    def check_split_balance(subset, name):
        labels = [subset.dataset.labels[idx] for idx in subset.indices]
        counts = Counter(labels)
        print(f"\n📊 {name} split class distribution:")
        for label, count in counts.items():
            print(f"Class {label}: {count} samples")

    check_split_balance(train_dataset, "Train")
    check_split_balance(val_dataset, "Validation")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model = ECGTransformer(
        input_channels=12, seq_len=1024, patch_size=50,
        embed_dim=48, num_heads=8, num_layers=6, num_classes=2,
        expansion=4, dropout=0.1
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = 150
    best_val_accuracy = 0.0
    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        scaler = GradScaler()
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Training (Epoch {epoch+1})", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with autocast():
                output = model(batch_X)
                loss = criterion(output, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        model.eval()
        val_accuracy = 0.0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                output = model(val_X)
                preds = output.argmax(dim=-1)
                val_accuracy += (preds == val_y).sum().item()
        
        val_accuracy /= len(val_dataset)
        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, model_folder, best=True)
        else:
            save_checkpoint(model, optimizer, epoch, model_folder)
    
    save_model(model_folder, model)
    print("Training complete!")

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.

def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'Tmodel.pth')
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")

    # Load the checkpoint dictionary
    checkpoint = torch.load(model_filename, map_location='cpu')
    print(checkpoint.keys())  # Should include: 'model_state_dict', 'epoch', etc.
    
    model = ECGHTransformer(num_channels=12, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])

    if verbose:
         print("Model loaded successfully from", model_filename)

    return model



def run_model(record, model, verbose):
    """
    Run inference on a single ECG record using the trained model.
    
    Args:
        record (str): Path to the ECG record.
        model (ECGHTransformer): Trained model instance.
        device (str): Device to run inference on ('cpu' or 'cuda').
        verbose (bool): Whether to print prediction details.
        
    Returns:
        binary_output (int): Predicted class (0 or 1).
        probability_output (float): Probability of class 1.
    """
    
    # 👉 Define device here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load and preprocess the ECG signal
    signal = load_ecg_signal(record, target_length=4096, target_leads=12)  # Adjust shape
    signal = np.expand_dims(signal.T, axis=0)  # Shape: (1, 12, 4096)

    # Convert to PyTorch tensor and move to device
    signal_tensor = torch.tensor(signal, dtype=torch.float32).to(device)

    # Ensure model is in evaluation mode
    model.to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        output = model(signal_tensor)
        probabilities = torch.softmax(output, dim=-1)  # Get class probabilities
        binary_output = torch.argmax(probabilities, dim=-1).item()
        probability_output = probabilities[0, 1].item()  # Probability of class 1

    if verbose:
        print(f"🔍 Prediction: {binary_output}, Probability: {probability_output:.4f}")

    return binary_output, probability_output


def save_model(model_folder, model):
    """
    Save the trained model to the specified folder.
    """
    os.makedirs(model_folder, exist_ok=True)  # Ensure the folder exists
    #model_filename = os.path.join(model_folder, 'model.sav')
    #joblib.dump({'model': model}, model_filename, protocol=0)
    model_filename = os.path.join(model_folder, 'Tmodel.pth')
    torch.save({'model_state_dict': model.state_dict()}, model_filename)
    print(f"💾 Model saved to {model_filename}")
