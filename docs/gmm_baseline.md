# GMM Baseline: One-Class SVM vs. Likelihood-Ratio Gate

See [Datasets & Protocol](datasets_and_protocol.md) for the shared MLAAD split used here.

## 1. Introduction

This is a baseline implementation from *Exploring the Synthetic Speech Attribution Problem Through Data-Driven Detectors* paper for open set source attribution. Two post hoc methods are used - a learned One-Class SVM (OCSVM) versus a simple top 2 log-likelihood-ratio (top1/top2) threshold. Unlike the other three experiments, it uses no pretrained neural embedding, establishing a floor for what pure generative density modeling achieves on raw spectral features.

## 2. Architecture

**Feature extraction.** Audio resampling to 16KHz and is converted to LFCCs:  30 ms frames with 59% overlap, 20 LFCC coefficients. Each clip becomes a variable-length `(T, 20)` sequence, cached to disk once (mirroring the protocol's split/language/model directory layout) and reused by both downstream scripts.

**Per-class GMMs.** One `sklearn.mixture.GaussianMixture` is fit per known class (24 total) on that class's pooled LFCC frames - 512 components, diagonal covariance is computed instead of whole matrix with `max_iter=400`, `reg_covar=1e-3`, fit in float64. Scoring a clip produces a 24-dim vector of mean per-frame log-likelihoods, one per class GMM; the closed-set prediction is always `argmax` over this vector, independent of the gate used.

**Two gates on identical GMMs.** 

The OCSVM path standard-scales the 24-dim score vector and fits one global RBF one-class SVM (`nu=0.1`, `gamma="scale"`) treating dense high-log-likelihood regions as "known"; its `decision_function` is the detection score. 

The ratio path needs no extra model: it takes the top1−top2 log-likelihood gap directly and declares "known" when the implied likelihood ratio `LR = exp(gap)` exceeds a threshold (default 2.0). Both gates share identical closed-set attribution, differing purely in the unknown-rejection rule.

## 3. Training Strategy

- 24 known classes fixed by protocol; one GMM per class (512 components, diagonal covariance), `reg_covar=1e-3`.
- Falls back to fewer components automatically if a class has fewer training frames than components.
- LFCC frames are cached to disk (`.npy`) once and reused across the OCSVM and ratio evaluation scripts.
- OCSVM: OneClassSVM fit once on training score vectors.
- Ratio gate: no training, a fixed threshold (`LR ≥ 2.0`) is applied directly to the log-likelihood gap, reusing the GMMs trained for the OCSVM path.
- Evaluated on the full eval split (n=33,791: 13,482 known / 20,309 unknown), broken out by the 4 seen/unseen quadrants.

## 4. Results

**Overall (eval, n=33,791)**

| Metric | OCSVM gate | Ratio gate (LR≥2.0) |
|---|---|---|
| AUROC | 0.5085 | 0.8857 |
| EER | 0.5018 | 0.1820 |
| macro-F1 (open, K+1) | 0.5728 | 0.6365 |
| Balanced detection accuracy | 0.5304 | 0.7056 |
| Unknown recall | 0.1415 | 0.4786 |
| Unknown F1 | 0.2368 | 0.6283 |
| Open-set accuracy (micro) | 0.4429 | 0.6546 |

**Per-quadrant open-set accuracy — ratio gate (OCSVM in parentheses)**

| Quadrant | n known/unknown | Open-set acc. | Macro-F1 (open) |
|---|---|---|---|
| lang_seen · model_seen | 8591 / 0 | 0.9085 (0.8697) | 0.6767 (0.6630) |
| lang_seen · model_not_seen | 91 / 7409 | 0.4933 (0.1385) | 0.0443 (0.0232) |
| lang_not_seen · model_seen | 4800 / 0 | 0.9406 (0.9463) | 0.1527 (0.1507) |
| lang_not_seen · model_not_seen | 0 / 12900 | 0.4729 (0.1482) | 0.0257 (0.0103) |

Note: Macro F1 here is taken across all models, so for 2nd and 4th quadrant with majority/full of unkown models with heavy imbalance in sample count is the reason for poor score.

**Insights**

1. The ratio gate massively outperforms the OCSVM gate for open set detection (AUROC 0.51→0.89, EER 0.50→0.18) despite sharing identical GMMs and closed-set accuracy - the decision rule matters more than the density model itself.
2. Closed set attribution is strong (96.6% accuracy)
3. The OCSVM gate sits near chance level (AUROC 0.51) - an RBF one class boundary fit on 24-dim per class log-likelihood vectors does not generalize to unseen model/language combinations.

## 5. Conclusion

1. A simple, training free top 2 log-likelihood ratio gate is a stronger and more practical open set decision rule than a separately learned OCSVM on this feature/model combination.
2. Per-class GMMs on LFCC features provide a strong closed set attribution floor (96.6%) that downstream neural approaches should be measured against.
3. Open-set generalization remains weak in the fully-unseen quadrant regardless of gate choice. Some classes/models didn't perform well in unkown - they strong correlate with known
