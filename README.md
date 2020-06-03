Installation
============

```bash
conda create -n neurolight
conda activate neurolight
conda install -c funkey pylp
conda install rtree
pip install -r requirements.txt
pip install .
```

Usage
=====

```python
from neurolight_evaluation.graph_metrics import Metric
from neurolight_evaluation.graph_score import score_graph

# Load in your predicted trees as nx.Graph's
predicted_tracings = load("predictions")

# Load in your ground truth trees as nx.Graph's
reference_tracings = load("predictions")

# Matching threshold: Given two nodes A in predictions and
# B in reference and distance d between them, the match_threshold
# determins the maximum d s.t. A and B may match
match_threshold = 5

# location_attr
location_attr = "location

# metric, what distance metric you want to use to compare these two trees
# Currently supports GRAPH_EDIT, ERL (expected run length), RECALL_PRECISION
metric = Metric.GRAPH_EDIT

# metric Kwargs: Some metrics need special arguments: i.e. GRAPH_EDIT distance
# needs to know how heavily you want to penalize false negatives.
metric_kwargs = {"node_spacing": 2, "details":True}

edit_distance, (split_cost, merge_cost, false_pos_cost, false_neg_cost) = score_graph(
    predicted_tracings=predicted_tracings,
    reference_tracings=reference_tracings,
    match_threshold = match_threshold,
    location_attr = location_attr,
    metric = metric,
    **metric_kwargs,
):
```