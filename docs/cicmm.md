# CICMM: Class-Conditional Prototype-Guided Mixture Model

See [Datasets & Protocol](datasets_and_protocol.md) for the shared MLAAD split used here.

## 1. Introduction

CICMM is a combination of cross entropy classification, supervised contrastive clustering, and a novel "ICMM" (inter class margin maximization) that pushes synthetic interpolated points away from class prototypes/centroids - explicitly modeling open set boundaries during training rather than relying purely on post-hoc methods. Per-class GMMs are fit on the resulting frozen embeddings and calibrated via conformal thresholds for accept/reject decisions. 


Different combination where experimented here: Two prototype-update strategies (batch vs. EMA - batch is in thesis) and two weighting schemes (manual architecture-family vs. auto centroid-distance - manual is in thesis) are compared across embedding dimensions (256/512 - 256D is in thesis) and GMM covariance modes (full/adaptive - full matrix is used in thesis).

## 2. Architecture

**Encoder.** XLS_R layer 5 frame features are projected, pooled with a learnable-query attention module (attention pooling), passed through a residual-block encoder (BatchNorm + GELU), and projected to a final L2-normalized embedding (256 or 512 dim) plus a linear 24-way classifier head. Once trained, the frozen encoder produces cached embeddings, logits, and labels for train/dev_known/dev_unknown/eval splits.

**Prototypes and weight matrix.** 
- A running centroid per class is tracked on the hypersphere, updated either as an **EMA** (momentum 0.99 - blends with history) or as a raw **batch** mean (no history, recomputed fully whenever the class appears). 
- A companion K×K class weight matrix encodes "repulsion pressure" between class pairs: the manual scheme assigns weight 3.0 to same architecture family pairs (e.g. Tacotron2 variants, XTTS variants, VITS variants, Bark variants) and 1.0 to cross-family pairs; the auto scheme instead derives weights inversely from learned centroid angular distances, recomputed once after a warm-up phase.

**Losses and GMM fitting.** The training loss combines cross-entropy, a supervised contrastive term (SupCon, temperature 0.05) on real embeddings, and the ICMM term, which penalizes synthetic points - interpolated from same batch embedding pairs under a curriculum-controlled blend ratio - for lying close to any class centroid, weighted by the pair weight matrix. 

1. Cross-entropy ($ \mathcal{L}_{CE} $)

Standard softmax classification loss on the 24-way logits from the real batch:

$$\mathcal{L}{CE} = -\frac{1}{B}\sum{i=1}^{B} \log \frac{\exp(\text{logit}{i,y_i})}{\sum{k} \exp(\text{logit}_{i,k})}$$

anchors the embedding space to the 24 known classes.

2. Supervised Contrastive loss ($\mathcal{L}_{SC}$)

On L2-normalized embeddings $z_i$, temperature $\tau=0.05$. Fo = {p \neq i : y_p = y_i}$ (same-class batch members), negatives = everyone else:

$$\text{sim}(i,j) = \frac{z_i \cdot z_j}{\tau}$$

$$\mathcal{L}{SC} = -\frac{1}{B}\sum{i} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(i,p))}{\sum_{a \neq i} \exp(\text{sim}(i,a))}$$

Pulls same class embeddings together and pushes different class ones apart on the hypersphere, this is what gives the embedding space enough structure for the per class GMMs fit later.

3. Synthetic "boundary" sample generation (not a loss — feeds into ICMM)

For a batch, uniformly pick an unordered class pair $(l_i, l_j)$ from the classes present, then a random member of each: $z_A$ (class $l_i$), $z_B$ (class $l_j$), and a mixing ratio $\omega \sim \mathcal{U}(\omega_{lo}, \omega_{hi})$:

$$z_{syn} = \frac{\omega z_A + (1-\omega) z_B}{\lVert \omega z_A + (1-\omega) z_B \rVert}$$

$\omega$ range is curriculum-controlled: $[0.45, 0.55]$ during the warm phase (epochs 1–20, tight interpolation near the midpoint), widening to $[0.2,
0.8]$ afterward (broader boundary coverage). Each synthetic pow_{syn} = W_{l_i, l_j}$ from the class weight matrix - class pair selection itself is uniform; only the loss contribution is weighted.

4. ICMM loss ($\mathcal{L}_{ICMM}$ — Inter-Class Margin Maximization)

Given the current class centroids $c_1,\dots,c_K$ (frozen for this batch — see centroid update below), for each synthetic point find its nearest centroid by cosine similarity and penalize closeness to it:

$$m(z_{syn}) = \max_k , (z_{syn} \cdot c_k)$$

$$\mathcal{L}_{ICMM} = \frac{\sum_s w_s , m(z_s)}{\sum_s w_s}$$

Minimizing this minimizes the maximum similarity to any centroid, i.e. it drives synthetic boundary points away from every known class.

After training, one 5 component GMM is fit per class on the frozen embeddings (full or adaptive covariance depending on per class sample count), with per class conformal negative log-likelihood thresholds calibrated on dev set.

## 3. Training Strategy

- Adam, lr 1e-3, weight decay 1e-4, 60 epochs, gradient clip 5.0, batch size 32, 200-frame (4s) crops.
- Curriculum: warm phase (epochs 1–20) uses tight synthetic-interpolation ratio ω∈[0.45,0.55]; the expansion phase widens to ω∈[0.2,0.8].
- Total loss = 0.5·CE + 0.5·SupCon + 0.5·ICMM; auto pair-weights (when used) are recomputed once, from learned centroids, at the end of the warm phase.
- Checkpoint selection by dev-known top-1 accuracy; `best.pt`/`last.pt` plus a `meta.json` of hyperparameters saved per run.
- 24 known classes, XLS_R layer 5, GMM fixed across all runs; embed_dim (256/512) × icmm_weighting (manual/auto) × gmm_covariance (full/adaptive) form the first version ablation grid (6 configs).
- The second version adds `centroid_mode` (batch vs. EMA) as an explicit, deliberately compared axis on top of the best first-version config (embed 512, manual weighting, full covariance) and class weight matrix is used in loss function rather than selection/sampling (done at first version).

Note: training stablizes very quickly before ~20 epoch

## 4. Results

**First-version** (eval n=33,791: 13,482 known / 20,309 unknown)

| Config | AUROC | EER | macro-F1 | F1 (unknown) | Known top-1 |
|---|---|---|---|---|---|
| e256_auto_full | 0.9160 | 0.1235 | 0.7329 | 0.6443 | 0.9706 |
| e256_manual_adaptive | 0.9627 | 0.0876 | 0.7237 | 0.6383 | 0.9664 |
| e256_manual_full | 0.9048 | 0.1299 | 0.7243 | 0.6434 | 0.9665 |
| e512_auto_adaptive | 0.9518 | 0.0894 | 0.7461 | 0.6653 | 0.9693 |
| e512_auto_full | 0.9156 | 0.1272 | 0.7452 | 0.6667 | 0.9691 |
| e512_manual_full | 0.8944 | 0.1473 | 0.7550 | 0.6772 | 0.9743 |

**Second-version, centroid-update ablation** (embed 512, manual, full covariance)

| centroid_mode | AUROC | EER | macro-F1 | F1 (unknown) | Known top-1 |
|---|---|---|---|---|---|
| batch | 0.9668 | 0.0878 | 0.7073 | 0.6173 | 0.9677 |
| ema | 0.9263 | 0.1158 | 0.7255 | 0.6505 | 0.9668 |

**Insights**

1. Adaptive covariance GMMs consistently give the best detection AUROC/EER (both `auto_adaptive` and `manual_adaptive` configs) at a small cost to macro F1 - full covariance overfits sparse classes.
2. Embedding dim 512 generally improves classification metrics (macro-F1, F1-unknown, known top-1) over 256 at matched settings, at the cost of slightly worse or comparable detection AUROC.
3. Batch mode centroid updates (2nd version) sharply improve detection over the 1st version baseline (AUROC 0.894→0.967, EER 0.147→0.088), but macro F1 drops (0.755→0.707) - a clear detection/classification trade-off from discarding prototype history.

## 5. Conclusion

1. Explicitly modeling class boundaries via synthetic interpolated points repelled from class prototypes (ICMM) is a viable way to inject open-set awareness directly into embedding training, rather than relying purely on post-hoc density estimation.
2. Making the prototype update strategy (batch vs. EMA) and pair weighting scheme (manual vs. auto) explicit, tracked axes was the key improvement between the first and second experimental versions.
3. No single configuration dominates on both detection and classification simultaneously. batch mode/adaptive covariance favor detection, while full covariance/higher embed_dim favor classification.
