# src/ravens_route/models_io.py

from pathlib import Path
import json
from typing import List, Optional, Dict

import xgboost as xgb

# __file__ = .../ravens_route/src/ravens_route/models_io.py
# parents[0] = .../ravens_route/src/ravens_route
# parents[1] = .../ravens_route/src
# parents[2] = .../ravens_route   <-- project root (where models/ lives)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PACKAGE_ROOT / "models"

# Cached globals so we only load once
_route_model: Optional[xgb.Booster] = None          # <-- Booster now
_route_features: Optional[List[str]] = None
_route_encoder: Optional[Dict[str, int]] = None


def get_route_features() -> List[str]:
    """
    Return the list of feature names that the route model expects.
    Loaded once from models/route_features.json.
    """
    global _route_features

    if _route_features is None:
        features_path = MODELS_DIR / "route_features.json"
        if not features_path.exists():
            raise FileNotFoundError(f"Could not find route_features.json at {features_path}")

        with open(features_path, "r") as f:
            _route_features = json.load(f)

        if not isinstance(_route_features, list):
            raise ValueError("route_features.json must contain a JSON list of feature names.")

    return _route_features


def get_route_encoder() -> Dict[str, int]:
    """
    Return mapping from route string -> encoded integer used in training.
    Loaded once from models/route_label_mapping.json.
    """
    global _route_encoder

    if _route_encoder is None:
        enc_path = MODELS_DIR / "route_label_mapping.json"
        if not enc_path.exists():
            raise FileNotFoundError(f"Could not find route_label_mapping.json at {enc_path}")

        with open(enc_path, "r") as f:
            _route_encoder = json.load(f)

        if not isinstance(_route_encoder, dict):
            raise ValueError("route_label_mapping.json must contain a JSON object {route: code}.")

    return _route_encoder


def get_route_model() -> xgb.Booster:
    """
    Return the trained XGBoost route model as a raw Booster.

    Loaded once from models/route_model.json using the native XGBoost API.
    """
    global _route_model

    if _route_model is None:
        model_path = MODELS_DIR / "route_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find route_model.json at {model_path}")

        booster = xgb.Booster()
        booster.load_model(str(model_path))
        _route_model = booster

    return _route_model
