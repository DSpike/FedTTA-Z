# Meta-Task vs Meta-Epoch: What's the Difference?

## 🎯 **Key Difference**

### **Meta-Task (`num_meta_tasks`):**

- **What**: Number of different learning tasks (episodes) created from the data
- **Purpose**: Each task is a separate few-shot learning episode
- **Unit**: Count of tasks (e.g., 20 tasks)

### **Meta-Epoch (`meta_epochs`):**

- **What**: Number of complete passes through ALL meta-tasks
- **Purpose**: How many times we iterate over all the tasks during training
- **Unit**: Number of iterations (e.g., 3 epochs)

---

## 📊 **Visual Explanation**

### **Training Structure:**

```
Client has: 20 meta-tasks (num_meta_tasks = 20)
            ↓
Meta-Epoch 1:
  ├─ Train on Task 1
  ├─ Train on Task 2
  ├─ Train on Task 3
  ├─ ...
  └─ Train on Task 20

Meta-Epoch 2:
  ├─ Train on Task 1 (again)
  ├─ Train on Task 2 (again)
  ├─ Train on Task 3 (again)
  ├─ ...
  └─ Train on Task 20 (again)

Meta-Epoch 3:
  ├─ Train on Task 1 (again)
  ├─ Train on Task 2 (again)
  ├─ Train on Task 3 (again)
  ├─ ...
  └─ Train on Task 20 (again)

Total Training: 20 tasks × 3 epochs = 60 training iterations
```

---

## 🔍 **Detailed Breakdown**

### **1. Meta-Task (`num_meta_tasks`)**

**Definition:**

- A meta-task is a single few-shot learning episode
- Each task contains:
  - **Support Set**: Examples for learning (k-shot per class)
  - **Query Set**: Examples for evaluation (n-query per class)

**Example:**

```python
# Configuration
num_meta_tasks = 20  # Create 20 different tasks

# Each task is created like this:
Task 1:
  - Support: 150 Normal + 150 Attack = 300 samples
  - Query: 10 Normal + 10 Attack = 20 samples

Task 2:
  - Support: 150 Normal + 150 Attack = 300 samples (different samples!)
  - Query: 10 Normal + 10 Attack = 20 samples (different samples!)

... (20 tasks total)
```

**Code Location:**

```python
# In create_meta_tasks function
for _ in range(n_tasks):  # n_tasks = num_meta_tasks
    # Create one task with support and query sets
    task = {
        'support_x': ...,
        'support_y': ...,
        'query_x': ...,
        'query_y': ...
    }
    meta_tasks.append(task)
```

**Purpose:**

- Provides diversity: Each task uses different samples
- Enables generalization: Model learns from multiple task distributions
- Mimics few-shot learning: Model learns to learn from few examples

---

### **2. Meta-Epoch (`meta_epochs`)**

**Definition:**

- One complete pass through all meta-tasks
- Similar to "epoch" in regular training, but for meta-learning
- In each epoch, the model sees all tasks again

**Example:**

```python
# Configuration
meta_epochs = 3  # Train for 3 epochs

# Training loop:
for epoch in range(meta_epochs):  # 3 iterations
    for task in meta_tasks:  # Loop through all 20 tasks
        # Train on this task
        loss = train_on_task(task)
        optimizer.step()
```

**Code Location:**

```python
# In meta_train method (models/transductive_fewshot_model.py)
def meta_train(self, meta_tasks, meta_epochs=100):
    for epoch in range(meta_epochs):  # Outer loop: epochs
        for task in meta_tasks:        # Inner loop: tasks
            # Train on each task
            support_loss = compute_loss(task['support_x'], task['support_y'])
            query_loss = compute_loss(task['query_x'], task['query_y'])
            total_loss = support_loss + query_loss
            optimizer.step()
```

**Purpose:**

- Repeated exposure: Model sees each task multiple times
- Better learning: More iterations = better convergence
- Standard practice: Similar to training epochs in regular ML

---

## 📈 **Comparison Table**

| Aspect                 | Meta-Task (`num_meta_tasks`)  | Meta-Epoch (`meta_epochs`)               |
| ---------------------- | ----------------------------- | ---------------------------------------- |
| **What it controls**   | Number of different tasks     | Number of training iterations            |
| **Default value**      | 20 tasks                      | 3 epochs                                 |
| **Optimization range** | 10-40 tasks                   | 2-5 epochs                               |
| **What changes**       | Different samples per task    | Same tasks, different training iteration |
| **Purpose**            | Task diversity                | Training depth                           |
| **Analogy**            | Number of different exercises | Number of times to practice              |

---

## 🎓 **Real-World Analogy**

Think of learning to solve math problems:

### **Meta-Tasks = Different Problem Sets**

- Problem Set 1: Algebra problems
- Problem Set 2: Geometry problems
- Problem Set 3: Calculus problems
- ... (20 different problem sets)

### **Meta-Epochs = Practice Rounds**

- Round 1: Solve all 20 problem sets (first attempt)
- Round 2: Solve all 20 problem sets again (second attempt - you're better now!)
- Round 3: Solve all 20 problem sets again (third attempt - even better!)

**Result:**

- 20 different problem sets (tasks)
- × 3 practice rounds (epochs)
- = 60 total practice sessions

---

## 🔢 **Total Training Iterations**

**Formula:**

```
Total Training Iterations = num_meta_tasks × meta_epochs
```

**Example with current config:**

```python
num_meta_tasks = 20  # 20 tasks
meta_epochs = 3      # 3 epochs

Total iterations = 20 × 3 = 60 training iterations per client per round
```

**With optimization ranges:**

- Minimum: 10 tasks × 2 epochs = **20 iterations**
- Maximum: 40 tasks × 5 epochs = **200 iterations**
- Current: 20 tasks × 3 epochs = **60 iterations**

---

## 📋 **In Your System**

### **Current Configuration:**

```python
# config.py
num_meta_tasks: int = 20    # 20 different tasks
meta_epochs: int = 3        # 3 training epochs
```

### **What Happens Per Client Per Round:**

1. **Create Tasks** (once):

   ```python
   meta_tasks = create_meta_tasks(
       data, labels,
       n_tasks=20  # Create 20 tasks
   )
   # Result: List of 20 task dictionaries
   ```

2. **Train for Multiple Epochs**:

   ```python
   model.meta_train(
       meta_tasks,      # 20 tasks
       meta_epochs=3    # Train for 3 epochs
   )
   # Result: Model sees each task 3 times
   ```

3. **Total Training**:
   - Task 1: Seen 3 times (once per epoch)
   - Task 2: Seen 3 times
   - ...
   - Task 20: Seen 3 times
   - **Total: 60 training iterations**

---

## ⚙️ **Optimization Impact**

### **Increasing `num_meta_tasks` (10 → 40):**

- **Pros:**
  - More task diversity
  - Better generalization
  - More varied learning scenarios
- **Cons:**
  - More training time (more tasks to process)
  - May need more data per client
  - Diminishing returns after a point

### **Increasing `meta_epochs` (2 → 5):**

- **Pros:**
  - Better convergence
  - Model sees same tasks multiple times
  - More learning iterations
- **Cons:**
  - More training time (more epochs)
  - Risk of overfitting
  - Diminishing returns

### **Trade-off:**

- **Few tasks, many epochs** (10 tasks × 5 epochs = 50 iterations)
  - Less diversity, but deep learning on each task
  - Good for limited data
- **Many tasks, few epochs** (40 tasks × 2 epochs = 80 iterations)
  - More diversity, but shallow learning per task
  - Good for abundant data

---

## ✅ **Summary**

| Concept              | Controls                              | Example     | Purpose           |
| -------------------- | ------------------------------------- | ----------- | ----------------- |
| **`num_meta_tasks`** | Number of different learning episodes | 20 tasks    | Task diversity    |
| **`meta_epochs`**    | Number of training passes             | 3 epochs    | Training depth    |
| **Together**         | Total training iterations             | 20 × 3 = 60 | Complete training |

**Key Takeaway:**

- **`num_meta_tasks`** = "How many different tasks?"
- **`meta_epochs`** = "How many times to practice?"

Both are now optimized by Optuna to find the best balance! ✅









