# Critical Evaluation: Behavioral Drift Detection for Online Exam Integrity

> **Verdict**: The pipeline is well-engineered and thoughtfully designed, with several things done correctly. However, there are **significant methodological concerns** that, while not invalidating the project, mean the current results should be presented with clear caveats. The results are **genuine but misleadingly optimistic** — not because of bugs, but because of the fundamental circularity of synthetic-only evaluation.

---

## 1. Result Validity — Are These Results Genuine?

### ✅ What's Done Right
- **Autoencoders trained on clean (normal-only) data** — This is correct practice for anomaly detection. The code in [main.py](file:///c:/Users/shhon/OneDrive/Desktop/ML/main.py#L240-L246) explicitly filters to `y_train == 0` before training.
- **Normalization stats computed on clean data before injection** — [main.py:L76-L78](file:///c:/Users/shhon/OneDrive/Desktop/ML/main.py#L76-L78) computes z-score stats before anomaly injection, preventing leakage.
- **Validation set also filtered to normal-only** for early stopping ([main.py:L250-L256](file:///c:/Users/shhon/OneDrive/Desktop/ML/main.py#L250-L256)), which is correct.
- **Per-model threshold optimization** — Each model gets its own threshold via [select_optimal_threshold](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/evaluate.py#105-162), not a shared one.
- **Robust z-scoring (median/MAD)** instead of mean/std for drift scores — good for skewed distributions.

### 🔴 Critical Concerns

#### 1a. Threshold Optimization on Validation Labels Is Circular
The threshold is tuned on validation data **that contains your own synthetic anomalies** using their **known labels** ([evaluate.py:L105-L161](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/evaluate.py#L105-L161)). This means the threshold is specifically calibrated to detect the exact types of anomalies you designed. In deployment, you don't know what anomalies look like — this is a form of **supervised information leaking into a nominally unsupervised task**.

> [!WARNING]
> The F1 scores are optimistic because they measure "how well can we detect the specific synthetic anomalies we designed, given that we can also tune a threshold against those same anomalies." A fixed percentile threshold (e.g., 95th percentile) would give a more honest assessment of unsupervised detection capability.

#### 1b. No Overfitting Detected, But No Evidence of Generalization Either
The results are plausible (F1 ≈ 0.61–0.62), not suspiciously perfect. Training losses aren't shown in the results directory, so I can't verify convergence curves, but the architecture (48 hidden, 16 latent, 2 layers) is tight enough that gross overfitting is unlikely. The **real concern isn't overfitting — it's underfitting to the wrong distribution** (synthetic vs. real cheating).

---

## 2. Synthetic Data Quality — Does It Resemble Real Cheating?

### ✅ What's Done Well
- **6 distinct anomaly types** covering timing, answer changes, navigation, and correlated shifts — better than most papers that use only 1-2 types.
- **Partial-session cheating** (Type 6, [preprocessing.py:L210-L264](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/preprocessing.py#L210-L264)) is the most realistic — students don't cheat uniformly through an entire exam.
- **Composition approach** (1 timing + optionally 1 supplementary, [preprocessing.py:L266-L298](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/preprocessing.py#L266-L298)) produces multi-dimensional anomalies, which is closer to reality.
- **Parametric variation** via `random.uniform` ranges instead of fixed multipliers adds natural variance.

### 🔴 Critical Concerns

#### 2a. Compression Factors Are Too Aggressive
The compression factors are `random.uniform(5.0, 10.0)`:
```python
# preprocessing.py:L37
compression = random.uniform(5.0, 10.0)
```
This means response times are **5–10x faster** than normal. In reality, cheating typically produces **1.5–3x** speedups (students still need to read the question, type an answer, etc.). A 10x compression transforms a 30-second response into 3 seconds — this is cartoonishly fast and trivially detectable by any model, including a simple threshold.

> [!IMPORTANT]  
> **This is the single biggest synthetic data concern.** With compressions this extreme, even `response_time_mean < X` as a single threshold would catch most anomalies. The model is succeeding partly because the anomalies are too extreme, not because the temporal modeling is capturing subtle behavioral shifts.

#### 2b. The Anomalies Are Feature-Aligned by Construction
Your anomalies directly manipulate the **same quantities** your features measure:
- Type 1 modifies timestamps → directly affects `response_time_mean`, `response_time_std`, [burst_ratio](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/feature_extraction.py#169-178), [speed_score](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/feature_extraction.py#161-168)
- Type 2 modifies answer changes → directly affects `answer_change_rate`
- Type 3 modifies timing → directly affects `pause_frequency`, `keystroke_rhythm_variance`

This creates a **tautological detection signal**: you're perturbing the features and then asking the model to detect perturbations in those same features. Real cheating would produce subtle, indirect signals — a student who copies answers might not change their timing pattern at all (they could copy at their natural pace from a phone).

#### 2c. Missing Cheating Modalities
Real-world cheating behaviors not covered:
- **Correct-answer-despite-speed**: A cheater who gets difficult questions right suspiciously fast (requires correctness data)
- **Collaboration patterns**: Two students with correlated answer sequences
- **Content-based signals**: Copy-paste text patterns, same uncommon wrong answers across students
- **Adaptive cheating**: Students who learn to mimic normal timing while cheating

---

## 3. Methodological Integrity — Pipeline Soundness

### ✅ Sound Decisions
- **Train/val/test split** uses same indices for all feature types — no alignment bugs
- **Masked MSE** for variable-length sequences in both training and evaluation
- **Packed sequences** in LSTM encoder for proper variable-length handling
- **Feature normalization on training data** applied to test data — no leakage here
- **Baselines trained on clean data only** (Isolation Forest, OC-SVM, Standard AE)

### 🔴 Red Flags

#### 3a. The Split Is Random, Not Chronological (Documentation Contradicts Code)

Your EXPLANATION.txt (Line 311-317) explicitly states:
> *"STEP 6: CHRONOLOGICAL TRAIN/VAL/TEST SPLIT (70/15/15). Per student, sessions are ordered by time..."*

But the actual code in [main.py:L104-L109](file:///c:/Users/shhon/OneDrive/Desktop/ML/main.py#L104-L109):
```python
np.random.seed(config['training']['seed'])
n = len(labels)
indices = np.random.permutation(n)
```

**The split is a random permutation, not chronological.** This is a significant discrepancy between the documentation and the implementation. Random splitting means:
- The same student's earlier sessions might appear in the test set while later sessions are in training
- This creates a **temporal leakage** risk where the model has seen "future" patterns from the same students during training

> [!CAUTION]
> If you present this as chronological splitting (as the paper implies), reviewers who check the code will flag this immediately. Either implement chronological splitting or correct the documentation.

#### 3b. Demographics Are Randomly Assigned — Fairness Analysis Is Decorative
From [data_loader.py:L238-L244](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/data_loader.py#L238-L244):
```python
# Random assignment for demonstration
np.random.seed(42)
oulad_sample = oulad_df.sample(len(sessions), replace=True).reset_index(drop=True)
```

OULAD demographics are **randomly sampled** and assigned to EdNet students. Since the demographic assignment is independent of behavioral features, fairness analysis is testing whether **random labels** produce equal flag rates — which they should by construction (unless the random seed creates accidental correlations). 

The fairness results showing `fair: True` across all attributes simply confirm that randomly assigned demographics are independent of detection scores. This doesn't validate that the fairness mechanism works against real demographic bias.

#### 3c. Anomaly Injection Before Splitting Creates Clean-Label Ambiguity
The pipeline injects anomalies into all sessions, then splits. The code correctly trains on clean-only data. However, the process means:
- Anomaly types are distributed randomly across train/val/test
- If certain anomaly types cluster in the test set by chance (especially at 20% contamination with 15% test), the test set may not be representative of the full anomaly distribution

A better approach: inject anomalies **only into the validation and test sets**, or verify the anomaly-type distribution across splits.

#### 3d. The "Personalized" Claim Is Not Implemented
The paper/explanation frames this as "personalized drift detection — comparing each student to themselves." But in the actual pipeline:
- There is **no per-student model fine-tuning**
- There is **no per-student baseline** computation
- The drift score uses **population-level** median/MAD normalization ([train.py:L222-L226](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/train.py#L222-L226))
- The EXPLANATION.txt describes lambda-weighted blending of personal and population scores (Section 3), but **this is never implemented in the code**

> [!WARNING]
> The project claims "personalized" detection but implements **population-level** anomaly detection. This is a fundamental gap between the narrative and the implementation. Either implement personalization or scale back the claims.

---

## 4. Result Benchmarking — Are These Results Genuinely Good?

### Summary of Results
| Model | F1 | ROC-AUC | PR-AUC | Precision | Recall |
|-------|-----|---------|--------|-----------|--------|
| **Transformer-AE** | 0.620 | **0.865** | **0.679** | 0.528 | 0.751 |
| **LSTM-AE (Ours)** | 0.614 | 0.859 | 0.626 | 0.513 | 0.766 |
| OneClassSVM | 0.590 | 0.781 | 0.605 | 0.577 | 0.603 |
| IsolationForest | 0.582 | 0.843 | 0.610 | 0.479 | 0.742 |
| StandardAutoencoder | 0.513 | 0.797 | 0.575 | 0.398 | 0.722 |
| RuleBased | 0.329 | 0.560 | 0.230 | 0.264 | 0.435 |

### Assessment

**The results are not suspiciously high** — in fact, they're moderate for anomaly detection with 20% contamination:
- F1 of 0.61 is modest. With known anomaly types and aggressive feature perturbation, higher F1 would be expected.
- **The precision is concerning**: ~0.51 means roughly half of flagged sessions are false alarms. This is unacceptable for real-world deployment where false accusations have serious consequences.
- The LSTM-AE barely beats traditional baselines. The margin over Isolation Forest is:
  - F1: +0.032 (5.5% relative improvement)
  - ROC-AUC: +0.016 (1.9% relative improvement)
  
  These margins are within random variation. Without confidence intervals or statistical significance tests, you cannot claim LSTM-AE is meaningfully better.

**The Transformer-AE actually outperforms the LSTM-AE**, which undermines the LSTM's "sequential temporal modeling" narrative. If the attention mechanism (which has no sequential inductive bias) works equally well, the temporal ordering may not be as informative as hypothesized.

> [!NOTE]
> For a LAK conference paper, these results would be acceptable IF properly contextualized. But presenting LSTM-AE as the "main contribution" when it doesn't clearly beat a Transformer-AE or even significantly beat Isolation Forest would raise reviewer eyebrows.

---

## 5. General Critical Observations

### 5a. The EXPLANATION.txt Contains Stale/Incorrect Numbers
Section 10 shows earlier results (LSTM-AE ROC-AUC: 0.612, IF: 0.824) that don't match the current [evaluation_results.yaml](file:///c:/Users/shhon/OneDrive/Desktop/ML/results/metrics/evaluation_results.yaml) (LSTM-AE ROC-AUC: 0.859, IF: 0.843). This suggests multiple iterations, which is fine — but make sure all documentation is consistent before submission.

### 5b. Ablation Study Is a Placeholder
[evaluate.py:L193-L217](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/evaluate.py#L193-L217) has [compute_ablation_study()](file:///c:/Users/shhon/OneDrive/Desktop/ML/src/evaluate.py#193-218) that returns empty dictionaries. Without actual ablation (removing features, varying architectures), you can't justify why 10 features, 48 hidden dims, or 16 latent dims are the right choices.

### 5c. No Statistical Significance Testing
Results are from a single train/test split with a fixed seed. Without cross-validation or multiple seeds, you can't distinguish signal from random variation. The LSTM-AE vs. IF difference (F1: 0.614 vs. 0.582) could easily flip with a different random seed.

### 5d. Missing Training Curves and Convergence Evidence
The `results/` directory doesn't contain training/validation loss curves. These are essential to verify the models converged properly and didn't overfit.

### 5e. Contamination Rate Mismatch
Config says `synthetic_contamination: 0.20` (20%), but [EXPLANATION.txt](file:///c:/Users/shhon/OneDrive/Desktop/ML/EXPLANATION.txt) says 10%. The config is what runs — so the actual contamination is 20%. At 20%, one-in-five sessions is anomalous, which is quite high and arguably makes detection easier than the 5-10% rates seen in real proctoring data.

---

## 6. Real-World Data Consideration

> **Your colleague is right.** Validating on real-world data is not just advisable — it's arguably necessary for the claims this project makes.

### Why Synthetic-Only Validation Is Insufficient

1. **You're measuring self-consistency, not detection ability.** You designed the anomalies, designed the features to detect those anomalies, and tuned the threshold on those anomalies. The entire evaluation loop is closed — there's no external validation that this pipeline would detect actual cheating.

2. **Real cheating is adversarial.** Cheaters who know behavioral monitoring is in place will deliberately try to maintain normal timing patterns. Your synthetic anomalies don't model this adversarial dynamic at all.

3. **The base rate problem.** In reality, cheating rates are estimated at 5-15% and the "difficulty" varies enormously. Your uniform 20% contamination with aggressive perturbations is not representative.

### Feasibility Assessment

| Real-World Validation Approach | Feasibility | Impact |
|-------------------------------|-------------|--------|
| Partner with university to get labeled proctoring data | Hard (privacy, IRB) | Very High |
| Use MOOC platform data with known integrity violations | Medium | High |
| Red-team study: ask students to simulate cheating | Medium | Medium-High |
| Expert review: present anomalous sessions to instructors for labeling | Easy | Medium |
| Domain shift test: train on EdNet, test on another clickstream dataset | Medium | Medium |

### Recommended Path
If real cheating labels are unavailable, the most feasible improvement is a **red-team study**: recruit volunteers to take practice exams under controlled conditions (honest vs. with cheat sheets), creating a small but genuine labeled dataset. Even 50-100 sessions would provide meaningful external validation.

---

## Summary: Key Actions Ranked by Priority

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 1 | **Implement or remove "personalized" claims** — code doesn't match narrative | 🔴 Critical | Medium |
| 2 | **Fix chronological split** — code uses random permutation, docs claim chronological | 🔴 Critical | Low |
| 3 | **Reduce compression factors** to 1.5–3x for realistic anomalies | 🟡 High | Low |
| 4 | **Add statistical significance** (multiple seeds or cross-validation) | 🟡 High | Medium |
| 5 | **Acknowledge fairness is on random demographics** in the paper | 🟡 High | Low |
| 6 | Add training curves to verify convergence | 🟡 Medium | Low |
| 7 | Implement actual ablation study | 🟡 Medium | Medium |
| 8 | Update stale numbers in EXPLANATION.txt | 🟢 Low | Low |
| 9 | Consider red-team study for external validation | 🟢 Bonus | High |

---

> **Bottom line:** This is solid engineering work with a well-structured pipeline, thoughtful architecture choices, and comprehensive evaluation apparatus. But the disconnect between the "personalized" narrative and the population-level implementation, combined with overly aggressive synthetic anomalies and the documentation/code inconsistency on splitting, would be caught by careful reviewers. Fix those issues and your project will be much stronger.
