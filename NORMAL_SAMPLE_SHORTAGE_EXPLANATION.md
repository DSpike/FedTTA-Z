# Why Clients Have Normal Sample Shortage Despite Sufficient Dataset Samples

## 🎯 **Key Question**

You have **56,000 Normal samples** in your dataset, but some clients end up with only **84-142 Normal samples** per meta-task. Why?

---

## ✅ **Answer: You're Looking at Per-Task, Not Per-Client Totals**

The logs you're seeing show **Normal samples per meta-task**, not **total Normal samples per client**.

### **What You're Seeing:**
```
Support set composition: 142 Normal (64.8%), 77 Attack (35.2%)
Support set composition mismatch: 101 Normal (56.7%) vs 77 Attack (43.3%)
```

**These are per-meta-task counts, NOT total client counts!**

---

## 📊 **Actual Distribution Process**

### **Step 1: Dataset Has Enough Samples**
- **Total Normal samples**: ~56,000 (or whatever your dataset has)
- **Total Attack samples**: ~107,000+ (combined from all attack types)

### **Step 2: Dirichlet Distribution Allocates to Clients**
With `dirichlet_alpha = 4.035` and 10 clients:
- Each client gets a **proportion** of Normal samples
- Example: Client 1 gets 12% = ~6,720 Normal samples
- Example: Client 2 gets 8% = ~4,480 Normal samples  
- Example: Client 10 gets 2% = ~1,120 Normal samples

**Each client has THOUSANDS of Normal samples total!**

### **Step 3: Meta-Tasks Sample from Client's Total**
When creating 35 meta-tasks per client:
- Each meta-task independently samples from the client's Normal samples
- **With replacement** - same samples can be used in multiple tasks
- Each task needs ~180 Normal samples (70% of 258 total support size)

### **Step 4: Why You See Low Numbers (84-142)**

The issue is **NOT** that clients lack Normal samples. The issue is:

1. **Client has, say, 5,000 Normal samples total** ✅ (enough!)
2. **Creating meta-task #1**: Randomly samples 180 Normal samples ✅
3. **Creating meta-task #2**: Randomly samples 180 Normal samples ✅
4. **...continues for 35 tasks...**

**BUT** - If you look at the **actual samples selected** for a specific task:
- The code samples from available Normal samples
- If it's sampling randomly and only finds 84-142 samples **that haven't been used in THIS specific task**, it uses those
- **But the client still has thousands more available for other tasks!**

---

## 🔍 **The Real Problem: Sampling Logic**

Looking at the code in `transductive_fewshot_model.py`:

```python
# 1. Add Normal samples
normal_mask = data_y == 0
normal_indices = torch.where(normal_mask)[0]  # ALL Normal samples for this client
if len(normal_indices) >= normal_shot:
    shuffled_normal = normal_indices[torch.randperm(len(normal_indices))][:normal_shot]
    support_x_list.append(data_x[shuffled_normal])
else:
    support_x_list.append(data_x[normal_indices])  # Use all available
```

**This should work correctly!** It samples from ALL Normal indices available to the client.

---

## 🤔 **Why Then Do We See Low Numbers?**

### **Hypothesis 1: Per-Task Sampling Without Proper Replacement**

If the code is somehow tracking "used" samples across tasks:
- Task 1 uses 180 Normal samples
- Task 2 can only use remaining samples
- After 20 tasks, only 84-142 Normal samples remain

**But this shouldn't happen** - each task should sample independently with replacement.

### **Hypothesis 2: Client Data Is Actually Limited**

Some clients might genuinely have very few Normal samples after Dirichlet distribution:
- With `dirichlet_alpha = 4.035`, minimum proportion could be ~2%
- 2% of 56,000 = 1,120 Normal samples
- If client has only 1,120 Normal samples total
- Creating 35 tasks × 180 Normal samples = 6,300 needed
- **This exceeds available samples!**

### **Hypothesis 3: Dataset Doesn't Actually Have 56,000 Normal Samples**

After preprocessing (zero-day split, filtering, etc.), the actual Normal samples might be much less:
- Original dataset: 56,000 Normal samples
- After preprocessing: Maybe only 20,000-30,000 Normal samples
- After Dirichlet: Some clients get only 400-600 Normal samples
- **This would explain the shortage!**

---

## 🎯 **How to Verify**

Check your actual dataset after preprocessing:

```python
# After preprocessing, before distribution
normal_mask = train_labels == 0
normal_count = normal_mask.sum().item()
print(f"Actual Normal samples after preprocessing: {normal_count:,}")
```

**If this number is much less than 56,000, that's the problem!**

---

## 💡 **Most Likely Cause**

The **most likely explanation** is:

1. **Original dataset**: Has 56,000 Normal samples
2. **After preprocessing** (zero-day split, filtering): Only ~15,000-25,000 Normal samples remain
3. **After Dirichlet distribution** (α=4.035, 10 clients):
   - Average: ~2,000 Normal samples per client
   - Minimum: ~300-600 Normal samples (if client gets 2-3%)
4. **Creating meta-tasks**:
   - Need 180 Normal samples per task
   - Client with 600 Normal samples can create ~3 tasks with 180 samples each
   - After that, only 60 samples remain → explains 84-142 samples in later tasks

---

## 🔧 **Solutions**

### **Solution 1: Increase Dirichlet Alpha** ⭐ RECOMMENDED
```python
dirichlet_alpha: float = 10.0  # More uniform distribution
```
**Effect**: Each client gets ~10% of Normal samples (more balanced)

### **Solution 2: Check Actual Normal Sample Count**
Verify how many Normal samples actually remain after preprocessing:
```python
# Add logging in preprocessing to see actual counts
print(f"Normal samples after preprocessing: {normal_count}")
```

### **Solution 3: Use More Normal Samples**
If preprocessing filters out too many Normal samples, adjust preprocessing to keep more.

### **Solution 4: Lower k_shot or support_normal_ratio**
```python
k_shot: int = 50  # Lower requirement per task
support_normal_ratio: float = 0.50  # Lower Normal proportion
```

---

## 📋 **Action Items**

1. **Add logging** to see actual Normal sample count after preprocessing
2. **Check Dirichlet distribution** to see minimum Normal samples per client
3. **Verify** that clients have enough samples before creating meta-tasks
4. **Consider increasing** `dirichlet_alpha` if clients genuinely lack samples

---

## ✅ **Conclusion**

You likely **DO have enough Normal samples in your dataset**, but:
- Preprocessing may reduce the count significantly
- Dirichlet distribution allocates them unevenly
- Some clients end up with insufficient samples for all 35 meta-tasks

**The solution is to either:**
1. Keep more Normal samples during preprocessing, OR
2. Increase `dirichlet_alpha` for more uniform distribution, OR  
3. Accept that some clients will create fewer meta-tasks (code handles this gracefully)










