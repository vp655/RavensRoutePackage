# ravens-route

A Python package that exposes a trained XGBoost route model to compute
P(catch | target, route-level features) for NFL passing plays.

## Basic usage

```python
from ravens_route import predict_route_prob
import pandas as pd

# df must have the same feature columns the model was trained on.
row = df.iloc[0]
p = predict_route_prob(row)
print(p)
