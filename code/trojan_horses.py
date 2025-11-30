# trojan_horses.py

import os
import pandas as pd

# Relative imports from the same package
from .MechanismAO import predict_antioxidant
from .ChemAO import ChemAO_pred

# =======================================================
# Resolve model paths INSIDE installed python package
# =======================================================

_PKG_DIR = os.path.dirname(__file__)

_MECH_MODEL_PATH = os.path.join(_PKG_DIR, "models", "best_model_state.pth")
_CHEM_MODEL_PATH = os.path.join(_PKG_DIR, "models", "ChemAO.joblib")

_INTERNAL_THRESHOLD = 0.5


class trojan_horses:
    @staticmethod
    def predict(smiles, api_key: str):
        """
        One-line inference wrapper.

        Public API:

            trojan_horses.predict(smiles, api_key)

        - Loads pretrained models from inside the package
        - Applies an internal fixed threshold for ChemAO
        - Returns a merged pandas DataFrame with:
            * BioChem-AOS (ChemAO) score
            * MA-AOS (MechanismAO) score
            * Pathway-level probabilities
        """

        # --------------------------
        # 1. MechanismAO (MA-AOS) inference
        # --------------------------
        mech_results = predict_antioxidant(
            smiles=smiles,
            api_key=api_key,
            model_path=_MECH_MODEL_PATH,
        )

        # --------------------------
        # 2. BioChem-AOS (ChemAO) inference
        # --------------------------
        chem_results = ChemAO_pred(
            smiles=smiles,
            api_key=api_key,
            model_path=_CHEM_MODEL_PATH,
            threshold=_INTERNAL_THRESHOLD,
            verbose=False,
        )

        # ensure dataframe format
        if isinstance(chem_results, dict):
            chem_df = pd.DataFrame([chem_results])
        else:
            chem_df = chem_results.reset_index(drop=True)

        # --------------------------
        # 3. Merge outputs
        # --------------------------
        rows = []

        for i, mech in enumerate(mech_results):
            chem_row = chem_df.iloc[i]

            row = {
                "SMILES": chem_row["SMILES"],
                "BioChem-AOS": float(chem_row["ChemAO Score"]),
                "BioChem-AOS_Label": int(chem_row["ChemAO Prediction"]),
                "MA-AOS": float(mech["Mechanism_AO"]),
            }

            # add pathway-wise probabilities from MechanismAO
            for name, prob in mech["targets"].items():
                row[f"{name}_prob"] = float(prob)

            rows.append(row)

        return pd.DataFrame(rows)





# # trojan_horses.py
# import pandas as pd
# from MechanismAO import *     # uses MechanismAO model :contentReference[oaicite:0]{index=0}
# from ChemAO import *                  # uses ChemAO model :contentReference[oaicite:1]{index=1}


# # trojan_horses.py

# import os
# # import pandas as pd

# # from MechanismAO import predict_antioxidant
# # from ChemAO import ChemAO_pred


# # =======================================================
# # Resolve model paths INSIDE installed python package
# # =======================================================

# _PKG_DIR = os.path.dirname(__file__)

# _MECH_MODEL_PATH = os.path.join(_PKG_DIR, "models", "best_model_state.pth")
# _CHEM_MODEL_PATH = os.path.join(_PKG_DIR, "models", "ChemAO.joblib")

# _INTERNAL_THRESHOLD = 0.5


# class trojan_horses:
#     @staticmethod
#     def predict(smiles, api_key: str):
#         """
#         One-line inference wrapper.

#         User-facing call:

#             trojan_horses.predict(smiles, api_key)

#         - Loads pretrained models from inside the package
#         - Applies fixed internal threshold
#         - Returns single merged DataFrame
#         """

#         # --------------------------
#         # 1. MechanismAOS inference
#         # --------------------------
#         mech_results = predict_antioxidant(
#             smiles=smiles,
#             api_key=api_key,
#             model_path=_MECH_MODEL_PATH,
#         )

#         # --------------------------
#         # 2. BioChemAOS inference
#         # --------------------------
#         chem_results = ChemAO_pred(
#             smiles=smiles,
#             api_key=api_key,
#             model_path=_CHEM_MODEL_PATH,
#             threshold=_INTERNAL_THRESHOLD,
#             verbose=False,
#         )

#         # ensure dataframe format
#         if isinstance(chem_results, dict):
#             chem_df = pd.DataFrame([chem_results])
#         else:
#             chem_df = chem_results.reset_index(drop=True)

#         # --------------------------
#         # 3. Merge outputs
#         # --------------------------
#         rows = []

#         for i, mech in enumerate(mech_results):
#             chem_row = chem_df.iloc[i]

#             row = {
#                 "SMILES": chem_row["SMILES"],
#                 "BioChem-AOS": float(chem_row["ChemAO Score"]),
#                 "ChemAO_Prediction": int(chem_row["ChemAO Prediction"]),
#                 "MA-AOS": float(mech["Mechanism_AO"]),
#             }

#             for name, prob in mech["targets"].items():
#                 row[f"{name}_prob"] = float(prob)

#             rows.append(row)

#         return pd.DataFrame(rows)



