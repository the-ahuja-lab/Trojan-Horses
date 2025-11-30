import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- METRICS IMPORTS ---
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, 
    cohen_kappa_score, matthews_corrcoef, 
    precision_score, recall_score, accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import warnings
import sys
import os
import random
import math
from torch.cuda.amp import autocast, GradScaler
import time

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")

# ============================================================================
# 0. CONFIGURATION & HARDCODED PARAMETERS
# ============================================================================

# !!! UPDATE THIS LIST WITH YOUR ACTUAL TARGET COLUMN NAMES !!!
# The model will ONLY train on these targets, in this exact order.
HARDCODED_TARGETS = ['HIF','Keap-1','NFkB','NOX','NRF2','Xanthine dehydrogenase']

CONFIG = {
    # --- BEST PARAMETERS APPLIED ---
    'lr': 2.8404776792848318e-05,
    'dropout_rate': 0.1640849839951254,
    'weight_decay': 0.0006039872793750558,
    'batch_size': 32,
    'hidden_arch': 'large', 
    
    # --- TRAINING CONFIG ---
    'epochs': 200,
    'patience': 100,
    'features_path': '../new_data/cdi_allbal.pkl',
    'labels_path': '../new_data/labels_allds.csv'
}

# ============================================================================
# 1. REPRODUCIBILITY UTILS
# ============================================================================

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    print(f"Random seed set to {seed}")

# ============================================================================
# 2. DATA PROCESSING
# ============================================================================

class DynamicDataProcessor:
    def __init__(self, target_list):
        # Uses the hardcoded list to define the hierarchy
        self.target_hierarchy = {
            "Antioxidant_System": target_list
        }

class MechanismInformedDataset(Dataset):
    def __init__(self, features, target_df, target_list):
        if hasattr(features, "toarray"): features = features.toarray()
        if hasattr(features, "values"): features = features.values
        
        # --- ENFORCE HARDCODED TARGETS ---
        # Select only the columns specified in target_list, in that order
        try:
            target_df_filtered = target_df[target_list]
        except KeyError as e:
            print(f"ERROR: One of the HARDCODED_TARGETS is not found in your CSV file: {e}")
            sys.exit(1)
            
        if hasattr(target_df_filtered, "values"): 
            target_df_vals = target_df_filtered.values
        else: 
            target_df_vals = target_df_filtered
            
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(target_df_vals)
        self.target_names = target_list 
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }

# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================

class MI_HMTL_Model(nn.Module):
    def __init__(self, input_dim, target_names, hidden_dims=[1024, 512, 256], dropout_rate=0.3):
        super(MI_HMTL_Model, self).__init__()
        self.input_dim = input_dim
        self.target_names = target_names
        self.dropout_rate = dropout_rate
        
        proc = DynamicDataProcessor(target_names)
        self.pathway_mapping = {} 
        self.pathway_names = list(proc.target_hierarchy.keys())

        # Shared Encoder
        self.shared_encoder = self._build_mlp(input_dim, hidden_dims[:-1], hidden_dims[-1])
        
        # Pathway Encoder
        self.pathway_encoders = nn.ModuleDict({
            pathway: self._build_mlp(hidden_dims[-1], [128, 64], 32)
            for pathway in self.pathway_names
        })
        
        self.target_heads = nn.ModuleDict()
        
        # Map targets
        for pname, targets in proc.target_hierarchy.items():
            for t in targets:
                self.pathway_mapping[t] = pname
        
        for target in self.target_names:
            pathway = "Antioxidant_System"
            if pathway in self.pathway_encoders:
                head_input_dim = hidden_dims[-1] + 32  
                self.target_heads[target] = self._build_mlp(
                    head_input_dim, [64, 32], 1, final_activation=None
                )

        # --- SCORE SCALER (FIXED INITIALIZATION) ---
        self.score_scaler = nn.Linear(1, 1)
        nn.init.constant_(self.score_scaler.weight, 2.0)
        nn.init.constant_(self.score_scaler.bias, 0.0)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def _build_mlp(self, input_dim, hidden_dims, output_dim, final_activation=None):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if final_activation:
            layers.append(final_activation)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        shared_repr = self.shared_encoder(x)
        shared_repr = self.dropout(shared_repr)
        
        pathway_reprs = {}
        for pathway_name, encoder in self.pathway_encoders.items():
            pathway_reprs[pathway_name] = encoder(shared_repr)
        
        predictions = {}
        all_probs_list = []

        for target_name, head in self.target_heads.items():
            pathway_name = "Antioxidant_System"
            if pathway_name in pathway_reprs:
                pathway_repr = pathway_reprs[pathway_name]
                combined_repr = torch.cat([shared_repr, pathway_repr], dim=1)
                
                logits = head(combined_repr).squeeze(-1)
                predictions[target_name] = logits
                all_probs_list.append(torch.sigmoid(logits))
        
        if all_probs_list:
            t_probs = torch.stack(all_probs_list, dim=1)
            # Average -> Scaler
            avg_prob = torch.mean(t_probs, dim=1).unsqueeze(1) 
            scaled_score = torch.sigmoid(self.score_scaler(avg_prob)).squeeze(1)
            antioxidant_score = scaled_score
        else:
            antioxidant_score = torch.zeros(x.size(0), device=x.device)
        
        return predictions, antioxidant_score

# ============================================================================
# 4. ROBUST WEIGHTED LOSS
# ============================================================================

class RobustWeightedLoss(nn.Module):
    def __init__(self, target_weights_dict):
        super(RobustWeightedLoss, self).__init__()
        self.target_weights = target_weights_dict
        
    def forward(self, target_predictions, target_actuals):
        device = next(iter(target_predictions.values())).device
        
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for name, pred_logits in target_predictions.items():
            if name in target_actuals:
                actual = target_actuals[name]
                mask = ~torch.isnan(actual)
                
                if mask.sum() > 0:
                    valid_pred = pred_logits[mask]
                    valid_actual = actual[mask]
                    
                    pos_wt = self.target_weights.get(name, 1.0)
                    pos_wt_tensor = torch.tensor(pos_wt, device=device)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        valid_pred, 
                        valid_actual, 
                        pos_weight=pos_wt_tensor,
                        reduction='mean'
                    )
                    
                    total_loss += loss
                    count += 1
                    
        if count > 0: total_loss /= count
        return total_loss

def calculate_pos_weights(df, target_list):
    weights = {}
    print("\n--- Dynamic Class Balancing ---")
    # Only calculate weights for the hardcoded targets
    for col in target_list:
        if col in df.columns:
            counts = df[col].value_counts()
            n_pos = counts.get(1.0, 1) 
            n_neg = counts.get(0.0, 1)
            
            weight = n_neg / n_pos
            weights[col] = weight
            print(f"Target: {col[:20]}... | Neg: {n_neg}, Pos: {n_pos} | Weight: {weight:.4f}")
        else:
            print(f"WARNING: Target {col} in HARDCODED_TARGETS but not in DataFrame!")
    return weights

def compute_comprehensive_metrics(predictions_dict, targets_dict):
    metrics = {}
    for name, preds in predictions_dict.items():
        if name in targets_dict:
            y_true = targets_dict[name].cpu().numpy()
            y_pred_logits = preds.cpu().numpy()
            y_pred_prob = 1 / (1 + np.exp(-y_pred_logits)) 
           
            mask = ~np.isnan(y_true)
            if mask.sum() > 0:
                y_true_valid = y_true[mask]
                y_pred_valid = y_pred_prob[mask]
                y_pred_bin = (y_pred_valid >= 0.5).astype(int)
                
                try:
                    if len(np.unique(y_true_valid)) > 1:
                        roc_auc = roc_auc_score(y_true_valid, y_pred_valid)
                    else:
                        roc_auc = 0.5
                except:
                    roc_auc = 0.5

                try:
                    acc = accuracy_score(y_true_valid, y_pred_bin)
                    bal_acc = balanced_accuracy_score(y_true_valid, y_pred_bin)
                    kappa = cohen_kappa_score(y_true_valid, y_pred_bin)
                    mcc = matthews_corrcoef(y_true_valid, y_pred_bin)
                    prec = precision_score(y_true_valid, y_pred_bin, zero_division=0)
                    rec = recall_score(y_true_valid, y_pred_bin, zero_division=0)
                    f1 = f1_score(y_true_valid, y_pred_bin, zero_division=0)
                    
                    metrics[name] = {
                        'accuracy': acc, 'balanced_accuracy': bal_acc, 'auroc': roc_auc,
                        'kappa': kappa, 'mcc': mcc, 'precision': prec, 'recall': rec, 'f1': f1
                    }
                except:
                    metrics[name] = {
                        'accuracy': 0, 'balanced_accuracy': 0, 'auroc': 0.5, 
                        'kappa': 0, 'mcc': 0, 'precision': 0, 'recall': 0, 'f1': 0
                    }
    return metrics

# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

def train_final_model(train_features, val_features, train_targets, val_targets, target_weights):
    
    arch_map = {
        "small": [512, 256, 128],
        "medium": [1024, 512, 256],
        "large": [2048, 1024, 512],
    }
    hidden_dims = arch_map[CONFIG['hidden_arch']]

    # Pass HARDCODED_TARGETS to dataset init
    train_dataset = MechanismInformedDataset(train_features, train_targets, HARDCODED_TARGETS)
    val_dataset = MechanismInformedDataset(val_features, val_targets, HARDCODED_TARGETS)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    
    model = MI_HMTL_Model(
        input_dim=train_features.shape[1],
        target_names=train_dataset.target_names,
        hidden_dims=hidden_dims,
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    # Use the optimized WEIGHT DECAY from config
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    criterion = RobustWeightedLoss(target_weights_dict=target_weights).to(device)
    scaler = GradScaler()

    best_val_auc = 0.0
    early_stop_counter = 0
    
    train_metrics_log = [] 
    val_metrics_log = []

    print(f"Starting training for {CONFIG['epochs']} epochs...")
    
    for epoch in range(CONFIG['epochs']): 
        # ==========================
        # 1. TRAINING PHASE
        # ==========================
        model.train()
        train_loss_accum = 0
        all_train_preds = {name: [] for name in train_dataset.target_names}
        all_train_targets = {name: [] for name in train_dataset.target_names}
        train_antiox_scores = [] 
        
        for batch in train_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            
            target_dict = {name: targets[:, i] for i, name in enumerate(train_dataset.target_names)}
            
            optimizer.zero_grad()
            with autocast():
                t_pred, antiox_score = model(features)
                loss = criterion(t_pred, target_dict)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_accum += loss.item()
            
            train_antiox_scores.extend(antiox_score.detach().cpu().numpy())
            
            for name in target_dict:
                all_train_preds[name].append(t_pred[name].detach().cpu())
                all_train_targets[name].append(target_dict[name].detach().cpu())
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        full_train_preds = {k: torch.cat(v) for k, v in all_train_preds.items()}
        full_train_targets = {k: torch.cat(v) for k, v in all_train_targets.items()}
        train_metrics = compute_comprehensive_metrics(full_train_preds, full_train_targets)
        
        for t_name, m_vals in train_metrics.items():
            row = {'epoch': epoch + 1, 'phase': 'train', 'target': t_name, 'loss': avg_train_loss, **m_vals}
            train_metrics_log.append(row)

        # ==========================
        # 2. VALIDATION PHASE
        # ==========================
        model.eval()
        val_loss_accum = 0
        all_val_preds = {name: [] for name in val_dataset.target_names}
        all_val_targets = {name: [] for name in val_dataset.target_names}
        val_antiox_scores = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)
                target_dict = {name: targets[:, i] for i, name in enumerate(val_dataset.target_names)}
                
                with autocast():
                    t_pred, antiox_score = model(features)
                    loss = criterion(t_pred, target_dict)
                val_loss_accum += loss.item()
                val_antiox_scores.extend(antiox_score.cpu().numpy())
                
                for name in target_dict:
                    all_val_preds[name].append(t_pred[name].cpu())
                    all_val_targets[name].append(target_dict[name].cpu())

        avg_val_loss = val_loss_accum / len(val_loader)
        mean_antioxidant_prob = np.mean(val_antiox_scores)
        
        full_val_preds = {k: torch.cat(v) for k, v in all_val_preds.items()}
        full_val_targets = {k: torch.cat(v) for k, v in all_val_targets.items()}
        val_metrics = compute_comprehensive_metrics(full_val_preds, full_val_targets)
        
        current_avg_val_auc = np.mean([m['auroc'] for m in val_metrics.values()])
        
        for t_name, m_vals in val_metrics.items():
            row = {
                'epoch': epoch + 1, 'phase': 'eval', 'target': t_name, 'loss': avg_val_loss,
                'antioxidant_score_avg': mean_antioxidant_prob, **m_vals
            }
            val_metrics_log.append(row)

        # ==========================
        # 3. CONSOLE & CHECKPOINTING
        # ==========================
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val AUC: {current_avg_val_auc:.4f} | Scaled Score: {mean_antioxidant_prob:.4f}")

        scheduler.step(current_avg_val_auc)

        if current_avg_val_auc > best_val_auc:
            best_val_auc = current_avg_val_auc
            early_stop_counter = 0
            
            torch.save({'state_dict': model.state_dict(), 'auc': best_val_auc}, 'best_model_state.pth')
            
            val_roc_data = {
                'epoch': epoch, 'best_auc': best_val_auc,
                'probs': {k: torch.sigmoid(v).cpu().numpy() for k, v in full_val_preds.items()},
                'targets': {k: v.cpu().numpy() for k, v in full_val_targets.items()},
                'antioxidant_scores': np.array(val_antiox_scores)
            }
            with open('best_val_roc_data.pkl', 'wb') as f: pickle.dump(val_roc_data, f)
                
            train_roc_data = {
                'epoch': epoch,
                'probs': {k: torch.sigmoid(v).cpu().numpy() for k, v in full_train_preds.items()},
                'targets': {k: v.cpu().numpy() for k, v in full_train_targets.items()},
                'antioxidant_scores': np.array(train_antiox_scores)
            }
            with open('best_train_roc_data.pkl', 'wb') as f: pickle.dump(train_roc_data, f)

            print(f"   >>> Best Model Saved (AUC: {best_val_auc:.4f})")
        else:
            early_stop_counter += 1
            
        if (epoch + 1) % 5 == 0:
            pd.DataFrame(train_metrics_log).to_csv('training_metrics_history.csv', index=False)
            pd.DataFrame(val_metrics_log).to_csv('validation_metrics_history.csv', index=False)

        if early_stop_counter >= CONFIG['patience']:
            print("   >>> Early stopping triggered.")
            break
    
    pd.DataFrame(train_metrics_log).to_csv('training_metrics_history.csv', index=False)
    pd.DataFrame(val_metrics_log).to_csv('validation_metrics_history.csv', index=False)
            
    return train_metrics_log, val_metrics_log, best_val_auc