# Clarification: Episodes vs Meta-Epochs in Your Setup

**Your Question**: "But the episode is 40 right in the configuration?"

**Answer**: You're confusing **training meta-epochs** with **evaluation episodes**. Let me clarify.

---

## Two Different Concepts

### 1. Meta-Epochs (Training) = 40

From your [config_loader.py:48](config_loader.py#L48):

```python
'meta_epochs': 40,  # Number of training episodes
'k_shot': 118,      # Support set size per episode
'n_query': 20,      # Query set size per episode
```

**What this means**:
- During **TRAINING**, you run **40 episodes** (meta-epochs)
- Each training episode has:
  - **Support set**: 118 samples (k_shot)
  - **Query set**: 20 samples (n_query)
- Total per training episode: 118 + 20 = **138 samples**

**Training loop**:
```python
for meta_epoch in range(40):  # 40 training episodes
    # Sample support set (118 samples)
    # Sample query set (20 samples)
    # Train on this episode
    # Repeat 40 times
```

### 2. Evaluation Episodes = 1 (Currently)

During **EVALUATION** (test time), you currently use:
- **1 single evaluation episode** with ~800 test samples
- This is completely separate from training

**Evaluation**:
```python
# After training is done, evaluate on test set
test_samples = ~800  # Single evaluation episode
base_model.evaluate(test_samples)
ttt_model.evaluate(test_samples)
```

---

## The Confusion

### Training (40 episodes)
```
Episode 1: 138 samples → train model
Episode 2: 138 samples → train model
...
Episode 40: 138 samples → train model
↓
Trained Model
```

### Evaluation (1 episode - THE PROBLEM)
```
Single test episode: ~800 samples → evaluate trained model
  - Worms: only 1 sample ❌
```

---

## What I'm Proposing

**Keep training as-is** (40 meta-epochs), but change **evaluation** to use multiple episodes.

### Current Evaluation (1 episode)
```python
# Single evaluation on ~800 test samples
test_set = stratified_sample(10000_pool, target_size=~800)
results = evaluate(test_set)
# Problem: Worms gets only 1 sample
```

### Proposed Evaluation (10 episodes)
```python
# Multiple evaluations, each on ~800 test samples
for episode in range(10):  # 10 EVALUATION episodes
    test_set = stratified_sample(full_82k_pool, target_size=~800)
    results[episode] = evaluate(test_set)

# Aggregate across 10 evaluation episodes
final_results = average(results)
# Benefit: Worms gets ~8-10 samples across episodes
```

---

## Why This Makes Sense

### Training vs Evaluation Are Different

| Aspect | Training (Meta-Epochs) | Evaluation (Episodes) |
|--------|----------------------|---------------------|
| **Purpose** | Learn to adapt quickly | Measure generalization |
| **Number of episodes** | 40 (your config) | 1 (current) → 10 (proposed) |
| **Episode size** | 138 samples (k_shot + n_query) | ~800 samples |
| **Data source** | Training set | Test set |
| **Goal** | Meta-learn prototypes | Evaluate zero-day detection |

### Why 40 Training Episodes is Fine

You need multiple training episodes to:
- Learn diverse attack patterns
- Build robust prototypes
- Enable fast adaptation (meta-learning goal)

### Why 1 Evaluation Episode is NOT Fine

With only 1 evaluation episode:
- Rare attacks (Worms) get 1 sample → unreliable
- No confidence intervals
- Sampling bias affects results
- Not publication-ready

---

## Your Actual Configuration

Looking at your config:

```python
# TRAINING CONFIGURATION
'meta_epochs': 40,        # Train for 40 episodes
'k_shot': 118,            # 118 samples per support set
'n_query': 20,            # 20 samples per query set

# This creates 40 training episodes of 138 samples each
# Total training iterations: 40 episodes
```

**These parameters control TRAINING, not evaluation.**

---

## What Actually Happens in Your Code

Let me trace through your evaluation:

### Training Phase (main.py)
```python
# Training: 40 meta-epochs
for meta_epoch in range(config.meta_epochs):  # 40 iterations
    support_set, query_set = sample_episode(train_data)
    # Train prototypical network
    # Update embeddings
    # Repeat for 40 episodes
```

### Evaluation Phase (main.py line 1117-1142)
```python
# Evaluation: SINGLE episode from test set
test_subset_size = min(10000, len(test_data))  # Limit pool
test_episode = stratified_sample(test_subset_size, target=~800)

# Evaluate once
base_results = evaluate_base_model(test_episode)  # Single episode!
ttt_results = evaluate_ttt_model(test_episode)    # Single episode!
```

**Problem**: Evaluation uses only **1 episode**, not 40.

---

## The Proper Solution

### Keep Training Configuration (40 meta-epochs)
```python
# config_loader.py - NO CHANGE
'meta_epochs': 40,  # Keep this for training
'k_shot': 118,
'n_query': 20,
```

### Add Evaluation Configuration
```python
# config_loader.py - ADD THIS
'eval_episodes': 10,           # Number of test episodes for evaluation
'eval_episode_size': 800,      # Samples per evaluation episode
'eval_pool_size': 82000,       # Use full test set pool (not 10k)
```

### Update Evaluation Code
```python
# In evaluation section
all_eval_results = []

for eval_episode in range(config.eval_episodes):  # 10 evaluation episodes
    # Sample from FULL test set (82k pool)
    test_episode = stratified_sample(
        pool_size=config.eval_pool_size,  # 82,000 samples
        target_size=config.eval_episode_size,  # 800 samples
        seed=SEED + eval_episode  # Different samples each time
    )

    # Evaluate on this episode
    base_results = evaluate_base_model(test_episode)
    ttt_results = evaluate_ttt_model(test_episode)

    all_eval_results.append({
        'episode': eval_episode,
        'base': base_results,
        'ttt': ttt_results
    })

# Aggregate results
final_results = aggregate_episodes(all_eval_results)
```

---

## Comparison Table

| Parameter | Training | Evaluation (Current) | Evaluation (Proposed) |
|-----------|----------|---------------------|----------------------|
| **Number of episodes** | 40 (meta_epochs) | 1 ❌ | 10 ✅ |
| **Samples per episode** | 138 (k_shot + n_query) | ~800 | ~800 |
| **Pool size** | Full training set | 10,000 subset ❌ | 82,000 full test set ✅ |
| **Data source** | Training set | Test set | Test set |
| **Worms samples per episode** | Varies | 1 ❌ | ~8-10 across episodes ✅ |
| **Total Worms samples** | - | 1 ❌ | ~80-100 ✅ |

---

## Why Your Confusion is Understandable

The terminology is confusing:

- **Meta-epochs** = Training episodes (40 in your config)
- **Evaluation episodes** = Not in your config (defaults to 1)

Both use "episodes", but they serve different purposes:
- Training episodes: Learn to adapt
- Evaluation episodes: Measure performance

---

## Summary

### Your Configuration

```python
'meta_epochs': 40,  # ← This is for TRAINING (40 training episodes)
'k_shot': 118,
'n_query': 20,
```

**This does NOT control evaluation episodes.**

### Current Evaluation

- **1 evaluation episode** with ~800 samples from 10k pool
- Worms: 1 sample ❌

### Proposed Evaluation

- **10 evaluation episodes** with ~800 samples each from 82k pool
- Worms: ~80-100 samples total ✅

### What to Change

**Training**: Nothing (keep meta_epochs=40)
**Evaluation**: Run 10 separate evaluation episodes, each with 800 samples drawn from full 82k test set

---

## Key Takeaway

- **meta_epochs=40** controls TRAINING (correct, keep it)
- **Evaluation episodes** are separate (currently 1, should be 10)
- These are two different concepts that happen to both use "episodes"

The solution is to add a loop in your evaluation code that runs 10 evaluation episodes (each with 800 samples from the full test set), then aggregates the results.

This does NOT change your training configuration at all.
