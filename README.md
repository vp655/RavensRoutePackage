# Ravens Route Package

A lightweight, production-ready Python package for evaluating wide receiver routes, generating catch-probability predictions, and animating route-level tracking data. The package ships with a pretrained XGBoost Booster model, a saved route label encoder, a feature specification JSON, and helper functions that make it easy to use the model on any row from your final matchup dataset.

# User Guide (Start Here)

## Installation

Install directly from GitHub via HTTPS:
pip install git+https://github.com/vp655/RavensRoutePackage.git

Or via SSH (if your SSH keys are configured with GitHub):
pip install git+ssh://git@github.com/vp655/RavensRoutePackage.git

This installs the package `ravens-route` (imported as `ravens_route` in Python) and all required dependencies.

## Predict Catch Probability

Example usage, assuming you have a `final_matchup_data.csv` built with the same feature engineering pipeline used to train the model:

import pandas as pd
from ravens_route import predict_route_prob

df = pd.read_csv("final_matchup_data.csv")
row = df.iloc[0]  # single play / matchup row

prob = predict_route_prob(row)
print("Catch Probability:", prob)

The function handles:
- Loading the pretrained XGBoost Booster.
- Loading the expected feature list.
- Loading the saved route label mapping.
- Encoding the route value (if given as a string).
- Selecting and ordering features correctly.
- Converting values to float32.
- Running the Booster’s predict() method and returning a float in [0, 1].

## Optional: Generate a Play Animation

If you are using the animation utilities to visualize a given play:

from ravens_route import animate_play_from_row
import pandas as pd

df = pd.read_csv("final_matchup_data.csv")
row = df.iloc[3]

anim = animate_play_from_row(
    row=row,
    data_dir="data_dir",              # folder containing tracking CSVs
    out_gif="animations/example.gif", # output GIF path
    fps=10,
    show=True
)

This will:
- Load the corresponding tracking data from data_dir.
- Create a frame-by-frame visualization of the route and coverage.
- Save the GIF to the specified path.
- Optionally display the animation if the environment supports it.

# Package Contents

After installation, the package is available under your Python 3.11 site-packages directory, for example:
C:\Users\light\AppData\Local\Programs\Python\Python311\Lib\site-packages\ravens_route\

The structure looks like:

ravens_route/
    __init__.py
    inference.py
    models_io.py
    models/
        route_model.json
        route_features.json
        route_label_mapping.json
    animation.py               # if included
    __pycache__/               # Python bytecode cache (ignored by git)

- route_model.json: XGBoost Booster model (trained route-level catch probability model).
- route_features.json: Ordered list of feature names used by the model.
- route_label_mapping.json: Mapping from route string (e.g. "GO", "SLANT", "WHEEL") to integer codes.

# How Prediction Works Internally

When you call predict_route_prob(row), the package does the following:

1. Uses models_io.get_route_model() to load a cached XGBoost Booster from ravens_route/models/route_model.json.
2. Uses models_io.get_route_features() to load the list of expected feature names.
3. Uses models_io.get_route_encoder() to load the route label mapping from route_label_mapping.json.
4. If row["route"] is a string, it encodes it into the correct integer code using the saved mapping (with a fallback for "undefined" if needed).
5. Subsets the row to exactly the features listed in route_features.json, in that exact order.
6. Converts all feature values to float32, raising a clear error if any NaNs appear.
7. Wraps the feature array in an xgboost.DMatrix and feeds it to the Booster.
8. Returns the scalar probability for the positive class (catch) as a Python float.

This design ensures:
- The same predictions across different machines.
- The same behavior between your development notebooks and the installed package.
- No dependency on ad-hoc notebook state or external preprocessing code.

# Dependency Versions

The package has been tested with and currently depends on the following versions (as reported by pip in the environment):

Core dependencies:
- pandas==2.2.0
- numpy==1.26.2
- xgboost==3.0.5

Visualization and image handling (used by animation utilities):
- matplotlib==3.8.2
- pillow==10.1.0

Transitive dependencies (automatically handled by pip):
- scipy==1.12.0 (from xgboost)
- python-dateutil==2.8.2
- pytz==2023.3.post1
- tzdata==2023.4
- contourpy==1.2.0
- cycler==0.12.1
- fonttools==4.47.0
- kiwisolver==1.4.5
- packaging==23.2
- pyparsing==3.1.1
- six==1.16.0

If you install via pip as shown above, these versions (or compatible ones) will be pulled in automatically. For maximum reproducibility, you can pin them explicitly in your own environment.

# Common Errors and Fixes

Below are the most likely issues you or other users may encounter, plus how to fix them.

1) Matplotlib / Animation Issues

Symptoms:
- Errors when generating GIFs or showing plots.
- Inconsistent or broken rendering of animations.
- Backend-related errors when calling animate_play_from_row.

Cause:
Often due to matplotlib version or backend issues.

Fix:
The package has been tested with matplotlib==3.8.2 and pillow==10.1.0. To enforce these versions, run:
pip install "matplotlib==3.8.2" "pillow==10.1.0"

Also ensure:
- You are not accidentally using a very old notebook or IDE backend that conflicts with this version.
- If needed, restart your kernel / Python interpreter after installation or upgrade.

2) macOS OpenMP / libomp Error (XGBoost)

Symptoms:
On macOS, you might see something like:
Library not loaded: @rpath/libomp.dylib

Cause:
XGBoost relies on OpenMP for parallelization. On macOS, you must install libomp manually.

Fix (macOS only):
brew install libomp
brew link libomp --force
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

Add the export line to your shell configuration (e.g., ~/.zshrc or ~/.bash_profile), then restart your terminal or IDE.

3) Python Version Incompatibility (Python 3.13+)

Symptoms:
ImportError involving xgboost or pandas internals, or strange behavior when installing xgboost on very new Python versions.

Cause:
Certain xgboost releases (including 3.0.5) are not yet compatible with Python 3.13+.

Fix:
Use Python 3.11 or Python 3.12 for this project. For example:
- On Windows: install Python 3.11 or 3.12 from python.org.
- On macOS/Linux: use pyenv, conda, or your package manager to create an environment with Python 3.11 or 3.12.

4) FileNotFoundError for route_model.json or feature JSON

Symptoms:
FileNotFoundError mentioning route_model.json, route_features.json, or route_label_mapping.json, usually with a path that does not contain models/.

Cause:
The models/ directory was not shipped with the installed package (e.g., incorrect packaging configuration), or the code is being run from a dev layout that doesn’t match the installed layout.

Fix (already handled in this package version, but for reference):
- Ensure the JSON files live inside src/ravens_route/models/ in the source repo.
- Ensure MANIFEST.in includes:
  recursive-include src/ravens_route/models *.json
- Ensure pyproject.toml includes:
  [tool.setuptools]
  include-package-data = true

In the installed environment, you can verify that the files exist with:
import ravens_route, pathlib
print(list((pathlib.Path(ravens_route.__file__).parent / "models").iterdir()))

5) Prediction Mismatch vs Old Notebook

Symptoms:
The probability from predict_route_prob(row) differs slightly from values computed in older notebooks.

Causes:
- The notebook used an XGBClassifier rather than the Booster.
- The notebook used a different feature subset or ordering.
- The notebook had different route encoding or preprocessing logic for some columns.

Why the package is authoritative:
- It always uses the saved Booster model from route_model.json.
- It always uses the saved feature ordering from route_features.json.
- It always uses the saved route label mapping from route_label_mapping.json.
As a result, if there is a mismatch, the package behavior should be taken as the canonical source of truth, and any notebook should be updated to mimic the package’s internal steps if exact reproduction is required.

# License

Insert your chosen license text here (e.g., MIT, BSD-3-Clause, or Apache-2.0), depending on how you intend others to use, modify, and distribute this package.

# Contact

For internal questions, extensions, or integration into other workflows, please contact:
Virochan Pandit
