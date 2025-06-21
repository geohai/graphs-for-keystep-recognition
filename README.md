# GLEVR: Graph Learning for Egocentric Video Recognition

This work won 1st place in the Ego-Exo4D Keystep Recognition Challenge at the EgoVis Workshop at CVPR 2025.  
Readme will be updated soon after CVPR 2025.  
Check out our lab website if interested [here](https://geohai.org/projects/spatiotemporal-graph-action.html).

**GLEVR** (Graph Learning on Egocentric Videos for keystep Recognition) is a lightweight, flexible graph-learning framework for fine-grained keystep recognition in egocentric videos. It leverages graph-based representations to capture long-term dependencies efficiently and integrates multi-view and multimodal data available only during training to boost performance at inference time.

---

## 🧠 Key Ideas

- **Node Classification for Keystep Recognition**: Each keystep segment is represented as a node in a temporal graph.
- **Multiview & Multimodal Training**: Additional exocentric views and video narrations are used to improve egocentric video understanding.
- **Efficient Graph Construction**: Sparse, flexible graph topologies yield high accuracy with significantly lower model size and compute cost than traditional video models.
- **Egocentric-Only Inference**: At test time, only the egocentric video view is used.

---

## 🧱 Architecture Overview

We support the following graph structures:

- **Egocentric Vision Graph**: A temporal graph with nodes per egocentric video clip.
- **Multiview Vision Graph**: Adds aligned exocentric clips as additional nodes with cross-view edges.
- **Heterogeneous Multimodal Graph**: Adds caption-based nodes using LLaMA3-generated segment summaries and LongCLIP embeddings.

---

## 🚀 Results

| Model                  | Narration | Val Acc | Test Acc |
|------------------------|-----------|---------|----------|
| TimeSFormer            | ❌        | 35.25   | 35.93    |
| EgoVLPv2 (EgoExo)      | ❌        | 38.21   | 38.69    |
| VI Encoder (EgoExo)    | ❌        | 40.23   | 41.53    |
| **MLE Baseline**       | ❌        | 40.40   | —        |
| **GLEVR (Ours)**       | ❌        | 54.69   | 52.36    |
| **GLEVR-Hetero (Ours)**| ✅        | **56.99** | **53.65** |

GLEVR outperforms all baselines on the Ego-Exo4D dataset with significantly smaller model size and compute footprint.

---

## 📊 Experimental Highlights

- **Long-Form Reasoning**: Performance improves >14% with full temporal context vs. isolated segments.
- **Multi-view Gains**: Using exocentric clips during training improves accuracy without increasing sample count.
- **Multimodal Alignment**: Automatically generated narrations boost performance via GLEVR-Hetero.

---

### Data

- **Dataset**: [Ego-Exo4D](https://ego-exo4d-data-url.org)
- **Visual features**: Omnivore Swin-L pretrained embeddings
- **Narrations**: Generated using [VideoRecap](https://github.com/your-forked-repo) + LLaMA-3 summaries

---

## 📚 Citation

If you find this work helpful, please consider citing our extended abstract:

```bibtex
@inproceedings{romero2025keystep,
  title     = {Keystep Recognition using Graph Neural Networks},
  author    = {Julia Lee Romero and Kyle Min and Subarna Tripathi and Morteza Karimzadeh},
  booktitle = {Extended Abstract, 2nd Workshop on Egocentric Perception (EPIC) at CVPR},
  year      = {2025},
  note      = {Presented at the 2nd Workshop on Egocentric Perception, CVPR 2025},
}
```
