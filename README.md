# Lab639_HAR
Human Action Recognition for Fisheye Dataset

Why fork repo?
- when I tried to modify any code, it may come arise differnet bugs and it hard to maintain.

# Ablation study

| Setting (CRL) | Setting (Triplet) | Setting (WCL) | Concat | Max | Mean | Sum |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✘ | ✘ | ✘ | 0.8870 | 0.7611 | 0.9018 | 0.9037 |
| **✔** | ✘ | ✘ | **0.9111** | 0.8463 | **0.9148** | **0.9093** |
| **✔** | **✔** | ✘ | 0.8759 | 0.8870  | 0.8888 | 0.9185 |
| **✔** | **✔** | ✘ | 0.8722 | 0.8796  | 0.8963  | 0.9018 |
| **✔** | ✘ | **✔** | 0.9092 | 0.8555 | 0.9092 | 0.9148 |

> **Note:** Rows with identical settings represent different implementation details. 

## EXP 1: reproduce results for `weighted contrastive loss`
I have been fixed the bug - `AttributeError: 'GraphModule' object has no attribute 'eval_graph'`.
- You need to modify `models/v3_model/baseline.py`

## EXP 2: reproduce results for `TripletMarginLoss` (similar to [DVANET](https://github.com/NyleSiddiqui/MultiView_Actions))
### Method 1
- see branch `TripletLoss`

Compared to [EXP 1](#exp-1-reproduce-results-for-weighted-contrastive-loss), just change `weighted CL` to `Triplet Margin Loss`.
### Method 2
- see branch `SA-DV_TripletMarginLoss`  

To implement `Same Action, Different View triplet margin loss`, it needs some modfications.  
1. add `action_features` (not fused features) as one of model return values.
2. To accomplish triplet loss, we need to find anchor, positive, negative.
    - positive: same action feature but different view 
    - negative: different action but same view
    - You can find another forwarding feature to get `same action` or `different action`.
    - To select view, we need to manipulate the `view dimension`.

## EXP 3: reproduce results for `remove either weighted_CL or triplet loss`
- see branch `Remove_CL`

Add document `#` for all related lines in code segments.

## EXP 4: reproduce results for `remove camera-to-region label`
- see branch `Remove_CameraID`

Initially, the program reads camera labels from a CSV file (preallocated with camera-to-region labels).  
However, the hidden `video_id` implicitly indicates which camera captured the video. $\rightarrow$ We can parse `video_id` to obtain the corresponding camera ID as the true camera label.