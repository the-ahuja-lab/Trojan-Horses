import os
import tempfile
from typing import Union, Iterable, List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from ChemicalDice import smiles_to_embeddings


def ChemAO_pred(
    smiles: Union[str, Iterable[str]],
    api_key: str,
    model_path: str,
    threshold: float = 0.5,
    smiles_col_name: str = "SMILES",
    fill_value: float = 0.0,
    verbose: bool = True,
):
    """
    Version WITHOUT fragmentation:
        - All aligned features are created as a NumPy array.
        - Only one DataFrame construction at the end.
    """

    # -----------------------------
    # 1. Normalize SMILES input
    # -----------------------------
    if isinstance(smiles, str):
        smiles_list: List[str] = [smiles]
        single = True
    else:
        smiles_list = list(smiles)
        single = False

    # -----------------------------
    # 2. ChemicalDice embedding
    # -----------------------------
    fd, temp_csv = tempfile.mkstemp(prefix="cdice_infer_", suffix=".csv")
    os.close(fd)
    pd.DataFrame({smiles_col_name: smiles_list}).to_csv(temp_csv, index=False)

    try:
        emb_raw = smiles_to_embeddings.collect_features_from_csv(
            filepath=temp_csv, key=api_key
        )
    finally:
        try: os.remove(temp_csv)
        except: pass

    emb_df = emb_raw if isinstance(emb_raw, pd.DataFrame) else pd.DataFrame(emb_raw)

    if smiles_col_name not in emb_df.columns:
        emb_df.insert(0, smiles_col_name, smiles_list)

    # -----------------------------
    # 3. Load model (joblib or xgb)
    # -----------------------------
    ext = os.path.splitext(model_path)[1].lower()
    model = None

    # joblib models
    if ext in (".joblib", ".pkl", ".sav"):
        model = joblib.load(model_path)
        if verbose:
            print(f"[INFO] Loaded sklearn/joblib model: {type(model)}")
    else:
        # Try load as xgboost first
        try:
            model_xgb = xgb.XGBClassifier()
            model_xgb.load_model(model_path)
            model = model_xgb
            if verbose:
                print("[INFO] Loaded XGBoost-native model.")
        except Exception:
            # fallback to joblib
            model = joblib.load(model_path)
            if verbose:
                print("[INFO] Loaded model via joblib load.")

    # -----------------------------
    # 4. Extract model feature names / count
    # -----------------------------
    model_feature_names = None
    n_features_in = None

    booster = None
    try:
        booster = model.get_booster()
        model_feature_names = booster.feature_names
    except:
        booster = None

    # sklearn attributes
    if model_feature_names is None:
        final_est = model
        if hasattr(model, "named_steps"):
            final_est = list(model.named_steps.values())[-1]
        
        if hasattr(final_est, "feature_names_in_"):
            model_feature_names = list(final_est.feature_names_in_)
        
        if model_feature_names is None and hasattr(final_est, "n_features_in_"):
            n_features_in = int(final_est.n_features_in_)

    # booster fallback
    if model_feature_names is None and n_features_in is None and booster is not None:
        try:
            n_features_in = booster.num_features()
        except:
            pass

    # -----------------------------
    # 5. Determine embedding column type
    # -----------------------------
    emb_cols = [c for c in emb_df.columns if c != smiles_col_name]

    emb_intlike = False
    try:
        emb_intlike = all(str(int(c)) == str(c) for c in emb_cols[:10])
    except:
        pass

    # -----------------------------
    # 6. Build list of expected model columns
    # -----------------------------
    if model_feature_names:
        normalized_model_cols = []
        for name in model_feature_names:
            # f0 → 0
            if isinstance(name, str) and name.startswith("f") and name[1:].isdigit():
                idx = int(name[1:])
                normalized_model_cols.append(idx if emb_intlike else name)
            elif isinstance(name, str) and name.isdigit() and emb_intlike:
                normalized_model_cols.append(int(name))
            else:
                normalized_model_cols.append(name)

        if verbose:
            print(f"[INFO] Using {len(normalized_model_cols)} model feature names.")

    elif n_features_in is not None:
        # take first N embedding columns
        if emb_intlike:
            sorted_int_cols = sorted(int(c) for c in emb_cols)
            normalized_model_cols = sorted_int_cols[:n_features_in]
        else:
            normalized_model_cols = emb_cols[:n_features_in]

        if verbose:
            print(f"[INFO] Using first {len(normalized_model_cols)} embedding columns.")

    else:
        # last fallback: use ALL embedding columns
        if emb_intlike:
            normalized_model_cols = sorted(int(c) for c in emb_cols)
        else:
            normalized_model_cols = emb_cols

        if verbose:
            print("[WARN] Using all embedding columns — no feature names available.")

    # -----------------------------
    # 7. Build aligned NUMPY array (FAST)
    # -----------------------------
    n = len(emb_df)
    m = len(normalized_model_cols)
    aligned_mat = np.full((n, m), fill_value, dtype=float)

    missing = []

    for j, col in enumerate(normalized_model_cols):
        if col in emb_df.columns:
            aligned_mat[:, j] = pd.to_numeric(emb_df[col], errors="coerce").fillna(fill_value).values
        else:
            # alt int<->str fallback
            if isinstance(col, int) and str(col) in emb_df.columns:
                aligned_mat[:, j] = pd.to_numeric(emb_df[str(col)], errors="coerce").fillna(fill_value).values
            elif isinstance(col, str) and col.isdigit() and int(col) in emb_df.columns:
                aligned_mat[:, j] = pd.to_numeric(emb_df[int(col)], errors="coerce").fillna(fill_value).values
            else:
                missing.append(col)

    if verbose and missing:
        print(f"[WARN] Missing {len(missing)} columns → filled with {fill_value}. Example: {missing[:5]}")

    # -----------------------------
    # 8. Predict
    # -----------------------------
    X = aligned_mat

    try:
        probs = model.predict_proba(X)[:, 1]
    except:
        preds = model.predict(X)
        probs = preds.astype(float)

    labels = (probs >= threshold).astype(int)

    # -----------------------------
    # 9. Build final output DataFrame
    # -----------------------------
    out = pd.DataFrame({
        smiles_col_name: smiles_list,
        "ChemAO Score": probs,
        "ChemAO Prediction": labels
    })

    return out.iloc[0].to_dict() if single else out