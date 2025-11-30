import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from ChemicalDice import smiles_to_embeddings

#=============================================================
#  HARD-CODED CONFIG (MUST MATCH TRAINING)
# ============================================================

HARDCODED_TARGETS = ['HIF','Keap-1','NFkB','NOX','NRF2','Xanthine dehydrogenase']

INPUT_DIM = 8192                      
HIDDEN_DIMS_LARGE = [2048, 1024, 512] 
DROPOUT_RATE = 0.1640849839951254    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
#  ARCHITECTURE (COPIED FROM YOUR TRAINING SCRIPT)
# ============================================================

class DynamicDataProcessor:
    def __init__(self, target_list):
        # Uses the hardcoded list to define the hierarchy
        self.target_hierarchy = {
            "Antioxidant_System": target_list
        }

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
            # Average -> Scaler (your original logic, NOT OR)
            avg_prob = torch.mean(t_probs, dim=1).unsqueeze(1) 
            scaled_score = torch.sigmoid(self.score_scaler(avg_prob)).squeeze(1)
            antioxidant_score = scaled_score
        else:
            antioxidant_score = torch.zeros(x.size(0), device=x.device)
        
        return predictions, antioxidant_score


# ============================================================
#  CHECKPOINT LOADER
# ============================================================

def load_trained_model(model_path="best_model_state.pth"):
    """
    Rebuilds the model exactly as in training and loads weights.
    """
    model = MI_HMTL_Model(
        input_dim=INPUT_DIM,
        target_names=HARDCODED_TARGETS,
        hidden_dims=HIDDEN_DIMS_LARGE,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)

    # your training: torch.save({'state_dict': model.state_dict(), 'auc': best_val_auc}, 'best_model_state.pth')
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # strip "module." if saved from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    if isinstance(ckpt, dict) and "auc" in ckpt:
        print(f"Loaded model from {model_path}")
    else:
        print(f"Loaded model from {model_path}")

    return model


# ============================================================
#  SMILES → EMBEDDINGS (ChemicalDice)
# ============================================================

def embed_smiles(smiles, api_key, temp_csv="temp_infer_smiles.csv"):
    """
    Takes a string or list of SMILES and returns a numpy array
    of shape (n_samples, INPUT_DIM) using ChemicalDice.
    """
    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = list(smiles)

    df = pd.DataFrame({"SMILES": smiles_list})
    df.to_csv(temp_csv, index=False)

    emb = smiles_to_embeddings.collect_features_from_csv(
        filepath=temp_csv,
        key=api_key
    )

    # cleanup temp CSV
    try:
        os.remove(temp_csv)
    except OSError:
        pass

    if hasattr(emb, "toarray"):
        emb = emb.toarray()
    elif hasattr(emb, "values"):
        emb = emb.values

    emb = np.asarray(emb, dtype=np.float32)

    if emb.shape[1] != INPUT_DIM:
        raise ValueError(f"Embedding dimension mismatch: expected {INPUT_DIM}, got {emb.shape[1]}")

    return emb


# ============================================================
#  MAIN INFERENCE FUNCTION
# ============================================================

def predict_antioxidant(
    smiles,
    api_key,
    model_path="best_model_state.pth"
):
    """
    Run inference for one or more SMILES.
    
    Returns a list of dicts (one per SMILES):
    {
        "antioxidant_score": float,   # your averaged+scaled score (NOT OR)
        "targets": {target_name: prob, ...}
    }
    """
    # 1. Load model
    model = load_trained_model(model_path)

    # 2. Get embeddings
    emb = embed_smiles(smiles, api_key)
    x = torch.tensor(emb, dtype=torch.float32).to(DEVICE)

    # 3. Forward pass
    model.eval()
    with torch.no_grad():
        t_pred, antiox_score = model(x)

    # 4. Collect results
    n_samples = x.shape[0]
    results = []

    for i in range(n_samples):
        # per-target probabilities
        targets_out = {}
        for t in HARDCODED_TARGETS:
            logit_i = t_pred[t][i]
            prob_i = torch.sigmoid(logit_i).item()
            targets_out[t] = float(prob_i)

        # antioxidant score from the model (already scaled)
        antiox_val = float(antiox_score[i].item())

        results.append({
            "Mechanism_AO": antiox_val,
            "targets": targets_out
        })

    return results
