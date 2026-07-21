# OSR Gate: Hyperbolic Busemann/Euclidean Embeddings + 7 Descriptor Logistic Gate

See [Datasets & Protocol](datasets_and_protocol.md) for the shared MLAAD split used here.

## 1. Introduction

This is the primary proposed approach: train a 24-way closed set classifier over XLS_R  features on the **official MLAAD split**, using either a hyperbolic Poincare ball head (Busemann loss) or a Euclidean cross entropy loss, then attach a lightweight 7 descriptor logistic regression "gate", trained separately - to accept (known) or reject (unknown) at inference. two XLS_R layers (1, 5) were planned to test how encoder depth affects the hyperbolic vs Euclidean comparison.

## 2. Architecture

**Shared trunk.** Both heads sit on an same attentive statistics pooling, over XLS_R_300m frames (D=1024): a mask aware attention weighting produces a weighted mean+std pooled vector, followed by BatchNorm. Only the embedding geometry differs downstream.

**Busemann (hyperbolic) head.** A linear layer maps the pooled vector into a 16 dim tangent space, which is then exponentially mapped into a point `z` on the Poincaré ball (curvature c=1). Classification uses 24 fixed, never trained "ideal" boundary prototypes placed via Riesz-energy repulsion on the sphere. The Busemann function `B_p(z) = log(‖p−z‖² / (1−‖z‖²))` measures distance to boundary prototype; the training loss adds a `−φ·log(1−‖z‖²)` penalty (φ=1.1) that pulls known class points toward, but not overlap the boundary, so embedding radius comes to encode prediction certainty. 

The Euclidean control head instead does linear embed/projection → L2-normalize → linear classifier, trained with plain cross entropy.

**7/6 descriptor gate.** After freezing the head, 7 signals are computed per sample (all oriented so higher = more known): softmax `p_max`, negative entropy, top1 top2 margin, logit energy (logsumexp), a "nearest-class" term (negative min Busemann distance, or negative L2 to class mean for Euclidean), negative Mahalanobis distance (Ledoit Wolf precision + per class means fit on train only), and  Busemann only Poincare radius `2·artanh(‖z‖)`. These are z scored and fed to a class balanced logistic regression fit purely on dev_known (accept) vs. dev_unknown (reject) - no eval leakage into either the embedding model or the gate.

## 3. Training Strategy

- AdamW, lr 1e-3, weight decay 1e-4, cosine annealing over 30 epochs, gradient clip 5.0, batch size 32, 4 second (200 frame) random crops for training, class balanced sampling over the 24 known classes.
- Checkpoint selection by dev known top 1 accuracy (margin free `Busemann` argmax, or CE-logit argmax for Euclidean).
- The gate is trained completely separately, after the embedding model is frozen: embeddings are cached once (train/dev_known/dev_unknown/eval), then logistic regression is fit only on dev_known/dev_unknown descriptor vectors.
- Uses the official MLAAD split (train_known / dev_known / dev_unknown / eval_all).
- Layer sweep intended over XLS_R layers {1, 2, 5}; only layers 1 and 5 have been trained and evaluated so far.
- Known result versioning caveat: the current `results/` no longer excludes a set of "overlap models" from the 41-category unknown count the way `results_1st/` did, so `results/`'s 41-category and 43-category detection numbers are now identical — `results_1st/` should be treated as the correct overlap-excluded reference.

## 4. Results

**Main comparison** (eval, n=33,791)

| Layer | Head | AUROC | EER | macro-F1 | Accuracy | Balanced Acc. | Closed top-1 |
|---|---|---|---|---|---|---|---|
| 1 | Busemann | 0.9473 | 0.1163 | 0.8770 | 0.8803 | 0.8832 | 0.9692 |
| 1 | Euclidean (CE) | 0.9687 | 0.0815 | 0.9140 | 0.9181 | 0.9562 | 0.9835 |
| 5 | Busemann | 0.9604 | 0.0978 | 0.8990 | 0.9026 | 0.9482 | 0.9835 |
| 5 | Euclidean (CE) | 0.9761 | 0.0597 | 0.9385 | 0.9403 | 0.9673 | 0.9886 |

**Descriptor ablation** (single descriptor AUROC vs. the learned gate)

| Descriptor | Bus. L1 | Bus. L5 | Eucl. L1 | Eucl. L5 |
|---|---|---|---|---|
| p_max | 0.9103 | 0.9413 | 0.9587 | 0.9663 |
| neg_entropy | 0.8963 | 0.9389 | 0.9622 | 0.9699 |
| margin | 0.9104 | 0.9391 | 0.9554 | 0.9633 |
| energy | 0.1953 | 0.1202 | 0.9669 | 0.9740 |
| near | 0.9251 | 0.9464 | 0.9615 | 0.9684 |
| neg_maha | 0.9264 | 0.9456 | 0.9523 | 0.9647 |
| radius | 0.8252 | 0.8977 | n/a | n/a |
| **learned gate** | **0.9473** | **0.9604** | **0.9687** | **0.9761** |

**Insights**

1. The Euclidean/CE beats the hyperbolic Busemann head on every headline metric at both layers tested, the hyperbolic geometry has not yet delivered its hypothesized OSR advantage in this setup.
2. Deeper features help both heads (layer 5 > layer 1).
3. The learned 7 descriptor gate outperforms every individual descriptor for both heads, confirming the descriptors carry complementary information worth combining.
4. "energy" is a near useless standalone descriptor for the Busemann head (AUROC 0.12–0.20, worse than chance inverted) but one of the strongest single descriptors for the Euclidean head (~0.97), a geometry specific quirk of unbounded hyperbolic logits.
5. The dev→eval AUROC gap ("memorization gap") is small (~0.008 – 0.012) for all four configs, indicating the gate generalizes well rather than overfitting to dev.

## 5. Conclusion

1. In the current runs, the Euclidean CE baseline is the stronger open set detector, so the hyperbolic Busemann approach has not yet demonstrated its intended benefit.
2. The 7 descriptor logistic gate is a robust, cheap, and reusable OSR mechanism that consistently beats any single descriptor regardless of embedding geometry.
3. Given the current gap, further tuning of φ/ball dimension or alternative hyperbolic formulations may be needed before the hyperbolic head can be favored over the simpler Euclidean control.
