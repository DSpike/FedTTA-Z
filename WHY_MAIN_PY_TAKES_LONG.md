# Why main.py Takes Long to Run - Performance Analysis

## Total Runtime Breakdown

Based on your recent run log (2025-12-09), the total runtime was approximately **4 minutes 17 seconds**:
- Started: 14:22:06
- Finished: 14:26:23

## Time Distribution

### 1. **Data Preprocessing: ~30 seconds** (12% of total time)
- Location: [preprocessing/blockchain_federated_cicids_preprocessor.py](preprocessing/blockchain_federated_cicids_preprocessor.py)
- Tasks:
  - Reading CSV files (CICIDS2017_train.csv, CICIDS2017_test.csv)
  - Feature extraction and normalization
  - Creating sequences (sequence_length=25, stride=12)
  - Train/test split
- **Impact**: Medium - necessary step, already optimized

### 2. **Meta-Training: ~3 minutes 45 seconds** (88% of total time) ⚠️ **BOTTLENECK**
- Location: [models/transductive_fewshot_model.py:2592-2729](models/transductive_fewshot_model.py#L2592-L2729)
- Configuration (from [config_loader.py:58-92](config_loader.py#L58-L92)):
  ```python
  meta_epochs: 22          # Number of training epochs
  k_shot: 200             # Support set samples per task
  n_query: 10             # Query set samples per task
  ```

#### Why Meta-Training is Slow:

**A. Number of Training Iterations**
- **22 epochs** × **number of meta-tasks per epoch**
- From logs: "Epoch 0" to "Epoch 20" took ~3 min 45 sec
- Average time per epoch: **~10.7 seconds**

**B. Per-Task Computational Cost** (each task in each epoch):
1. **Transductive Inference** (most expensive):
   - Lines 2648-2652 in transductive_fewshot_model.py
   - Performs label propagation and prototype refinement
   - Processes 200 support samples + 10 query samples

2. **Forward Passes**:
   - Support embeddings: `self.extract_embeddings(support_x)` - 200 samples
   - Query embeddings: `self.extract_embeddings(query_x)` - 10 samples
   - Distance computations: `torch.cdist()` for all samples vs prototypes

3. **Backward Pass & Optimization**:
   - Gradient computation through entire model
   - Gradient clipping
   - Optimizer step (AdamW)

**C. Running on CPU** ⚠️ **MAJOR SLOWDOWN**
- Your system runs on CPU (no CUDA available)
- CPU processing is **10-100x slower** than GPU for deep learning
- Mixed precision disabled (FP16 only works on GPU)

### 3. **TTT Adaptation: ~7 seconds** (3% of total time)
- Location: Test-time training adaptation
- From logs: 6 TTT runs, each taking ~5-6 seconds
- Steps per TTT run: 83 steps (from ttt_base_steps=194, reduced by complexity)

### 4. **Evaluation & Visualization: ~3 seconds** (1% of total time)
- Confusion matrices, ROC curves, PR curves
- Performance metrics computation
- Plot generation

---

## Main Performance Bottlenecks (Ranked)

### 🔴 **Critical Bottleneck #1: CPU Processing**
- **Impact**: 10-100x slower than GPU
- **Evidence**: Meta-training takes 3m 45s for just 22 epochs
- **Solution**:
  ```bash
  # Use GPU if available (NVIDIA GPU with CUDA)
  python main.py --dataset CICIDS2017
  ```
- **Expected speedup**: 10-50x faster with GPU

### 🟡 **Bottleneck #2: High Number of Meta-Epochs (22)**
- **Impact**: Directly proportional to runtime
- **Current**: 22 epochs × 10.7 sec = ~235 seconds
- **Solution**: Reduce meta_epochs for faster testing
  ```python
  # In config_loader.py line 65:
  'meta_epochs': 10,  # Reduce from 22 → 10 (2.2x faster)
  ```
- **Trade-off**: Slightly lower accuracy (but may not be noticeable)

### 🟡 **Bottleneck #3: Large Support Set Size (k_shot=200)**
- **Impact**: More samples = more forward passes + larger distance matrices
- **Current**: 200 support samples per task
- **Solution**: Reduce k_shot for faster training
  ```python
  # In config_loader.py line 66:
  'k_shot': 100,  # Reduce from 200 → 100 (1.5x faster per task)
  ```
- **Trade-off**: May reduce few-shot learning quality

### 🟢 **Bottleneck #4: Transductive Inference Overhead**
- **Impact**: Label propagation and prototype refinement add computational cost
- **Current**: Runs for every task in every epoch
- **Solution**: Could disable for faster baseline testing
- **Trade-off**: Loses the main advantage of your transductive approach

---

## Optimization Recommendations

### **Quick Wins (No Code Changes):**

1. **Use GPU if available** (10-50x speedup)
   ```bash
   # Check if GPU is available:
   python -c "import torch; print(torch.cuda.is_available())"

   # If True, your code will automatically use GPU
   python main.py --dataset CICIDS2017
   ```

2. **Reduce meta_epochs for development/testing** (2.2x speedup)
   - Edit [config_loader.py:65](config_loader.py#L65)
   - Change `'meta_epochs': 22` → `'meta_epochs': 10`
   - Use 22 epochs only for final experiments

3. **Reduce k_shot during development** (1.5x speedup)
   - Edit [config_loader.py:66](config_loader.py#L66)
   - Change `'k_shot': 200` → `'k_shot': 100`
   - Use 200 only for final experiments

### **Medium Wins (Minor Code Changes):**

4. **Enable Early Stopping** (already implemented but could be more aggressive)
   - The meta-training loop could stop early if loss plateaus
   - Add to line 2720 in transductive_fewshot_model.py:
   ```python
   if epoch > 5 and abs(avg_loss - training_history['epoch_losses'][-2]) < 0.001:
       logger.info(f"Early stopping at epoch {epoch} (loss plateau)")
       break
   ```

5. **Reduce Logging Verbosity**
   - Change log level to WARNING during training
   - Add to main.py before training:
   ```python
   logging.getLogger('models.transductive_fewshot_model').setLevel(logging.WARNING)
   ```

6. **Use Batch Processing for Meta-Tasks**
   - Currently processes tasks one-by-one (line 2638)
   - Could batch multiple tasks together (more complex)

---

## Expected Runtime After Optimizations

| Configuration | Meta-Epochs | k_shot | Device | Expected Runtime |
|---------------|-------------|---------|--------|------------------|
| **Current** | 22 | 200 | CPU | ~4 min 17 sec |
| Quick Win #2 | 10 | 200 | CPU | ~2 min 15 sec |
| Quick Win #3 | 22 | 100 | CPU | ~3 min 5 sec |
| Quick Win #2+#3 | 10 | 100 | CPU | ~1 min 30 sec |
| **Best (GPU)** | 22 | 200 | GPU | **~10-30 seconds** ⚡ |

---

## Code Locations Summary

1. **Meta-training loop**: [models/transductive_fewshot_model.py:2630-2729](models/transductive_fewshot_model.py#L2630-L2729)
2. **Config (meta_epochs)**: [config_loader.py:65](config_loader.py#L65)
3. **Config (k_shot)**: [config_loader.py:66](config_loader.py#L66)
4. **Config (ttt_base_steps)**: [config_loader.py:82](config_loader.py#L82)
5. **Preprocessing**: [preprocessing/blockchain_federated_cicids_preprocessor.py](preprocessing/blockchain_federated_cicids_preprocessor.py)

---

## Conclusion

**Main Reason**: Your system runs on **CPU**, and meta-training with **22 epochs** × **200 samples per task** is computationally expensive.

**Fastest Solution**: Use a GPU (10-50x speedup)

**If GPU not available**: Reduce `meta_epochs` to 10 and `k_shot` to 100 for development (2-3x speedup)

The 4-minute runtime is actually **reasonable for CPU-based deep learning** with these parameters. Most researchers use GPUs for this type of work, which would reduce the runtime to **10-30 seconds**.
