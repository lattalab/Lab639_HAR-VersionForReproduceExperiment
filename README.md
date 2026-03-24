# Lab639_HAR
Human Action Recognition for Fisheye Dataset

Why fork repo?
- when I tried to modify any code, it may come arise differnet bugs and it hard to maintain.

## EXP 1: reproduce results for `weighted contrastive loss`
I have been fixed the bug - `AttributeError: 'GraphModule' object has no attribute 'eval_graph'`.
- You need to modify `models/v3_model/baseline.py`