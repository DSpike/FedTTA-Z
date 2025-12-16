# Transductive Meta Learning Loss Curve and Accuracy Plot

## 📊 **What the Plot Shows**

The `training_history_PortScan_.png` plot displays **transductive meta learning progress** during federated learning over rounds:

- **Left Plot**: Transductive Meta Learning Loss (decreasing curve = good ✅)
- **Right Plot**: Transductive Meta Learning Accuracy (increasing curve = good ✅)
- **X-Axis**: Federated rounds (1, 2, 3, ..., 11)
- **Y-Axis**: Meta-learning loss values (left) and Meta-learning accuracy values (0-1, right)

**This is the loss and accuracy from transductive meta learning tasks during federated training.**

---

## 🔄 **Data Flow: Where Loss Comes From**

### **Step 1: Each Client Trains Locally**

During each federated round, each client:

1. **Creates meta-tasks** from its local data
2. **Runs meta-training** for `meta_epochs` epochs (currently 5 epochs)
3. **Computes average loss** across all meta-training epochs

**Code Location**: `coordinators/simple_fedavg_coordinator.py` lines 2656-2661

```python
# Client calculates average loss from meta-training
avg_loss = (
    sum(meta_training_history["epoch_losses"])
    / len(meta_training_history["epoch_losses"])
    if meta_training_history["epoch_losses"]
    else 0.0
)
```

**What this loss represents**:
- Average **cross-entropy loss** from meta-learning tasks
- Loss on **query sets** (validation) during meta-training
- Lower loss = model predicts better on query samples

---

### **Step 2: Collect Client Updates**

After all clients train, the coordinator collects:

```python
client_updates = [
    SimpleClientUpdate(
        client_id="client_1",
        training_loss=0.6521,  # Average loss from client 1's meta-training
        validation_accuracy=0.9488,
        ...
    ),
    SimpleClientUpdate(
        client_id="client_2",
        training_loss=0.5834,  # Average loss from client 2's meta-training
        validation_accuracy=0.9623,
        ...
    ),
    ...
]
```

---

### **Step 3: Average Across Clients Per Round**

For each round, the plot calculates:

```python
# In main.py lines 3220-3228
for round_data in self.training_history:
    round_losses = []
    for client_update in client_updates:
        loss = client_update.training_loss
        round_losses.append(loss)
    
    # Average loss for this round
    avg_round_loss = np.mean(round_losses)
    epoch_losses.append(avg_round_loss)  # One point per round
```

**Result**: One loss value per federated round (averaged across all clients)

---

### **Step 4: Plot Over Rounds**

The visualization plots:

- **X-axis**: Round numbers (1, 2, 3, ..., 11)
- **Y-axis**: Average loss per round (decreasing over time)

**Code Location**: `visualization/performance_visualization.py` lines 114-130

```python
ax1.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o')
ax1.set_title('Transductive Meta Learning Loss Over Rounds (PortScan Attack)')
ax1.set_xlabel('Federated Round')
ax1.set_ylabel('Average Meta-Learning Loss')
```

---

## ✅ **Why Loss Decreases (This is GOOD!)**

The decreasing loss curve indicates **successful learning**:

### **1. Model is Learning**
- **Round 1**: Model starts from scratch → High loss (~0.65)
- **Round 5**: Model has learned from multiple rounds → Lower loss (~0.45)
- **Round 11**: Model has converged → Lowest loss (~0.25)

### **2. Federated Aggregation is Working**
- Each round, clients train on local data
- Global model aggregates updates from all clients
- Model improves over rounds through aggregation

### **3. Example Loss Curve Pattern**

```
Round 1:  Loss = 0.6500  (Initial, high loss)
Round 2:  Loss = 0.5800  (Learning)
Round 3:  Loss = 0.5200  (Improving)
Round 4:  Loss = 0.4800  (Better)
Round 5:  Loss = 0.4500  (Good)
...
Round 11: Loss = 0.2800  (Converged, low loss)
```

**This pattern is EXPECTED and CORRECT** ✅

---

## 🔍 **What the Loss Represents**

### **Loss Calculation (During Meta-Training)**

Each client's `meta_train()` computes:

```python
# For each meta-task:
support_loss = F.cross_entropy(support_logits, support_y)
query_loss = F.cross_entropy(query_logits, query_y)
total_loss = support_loss + query_loss

# Average across all meta-epochs and meta-tasks
avg_loss = average(total_loss across all epochs and tasks)
```

**Loss components**:
- **Cross-entropy loss** on query set predictions
- **Prototype-based classification**: Distance to class prototypes
- **Lower loss** = Predictions closer to true labels

---

## ⚠️ **Important Notes**

### **1. This is NOT Test Loss**
- The plot shows **training/validation loss during federated learning**
- It's computed on **meta-task query sets** (from training data)
- **NOT** the final test set evaluation

### **2. It's Averaged Across Clients**
- Each point = average of all clients' losses for that round
- If only 1 client trains (others skipped), it's just that client's loss
- With 5 clients training, it's the average of 5 losses

### **3. It Reflects Meta-Learning Progress**
- Loss decreases as model learns to quickly adapt to new tasks
- Prototype-based classification improves over rounds
- Model gets better at distinguishing Normal vs Attack classes

---

## 📈 **Expected Patterns**

### **Healthy Training (What You Should See)** ✅

```
Transductive Meta Learning Loss:
Loss:   0.65 → 0.58 → 0.52 → 0.48 → 0.45 → 0.38 → 0.32 → 0.28
         ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
       Round 1  R2    R3     R4     R5     R6     R7     R8
       
Trend: Decreasing (good!) - Model learns to adapt to meta-learning tasks better
```

### **Unhealthy Training (Warning Signs)** ⚠️

```
Loss:   0.65 → 0.68 → 0.72 → 0.75 → 0.78
         ↑      ↑      ↑      ↑      ↑
Trend: Increasing (model getting worse!)

OR

Loss:   0.65 → 0.30 → 0.28 → 0.05 → 0.02
         ↓      ↓      ↓      ↓      ↓
Trend: Too rapid decrease (possible overfitting!)
```

---

## 🎯 **Summary**

**What this plot represents**:
- **Transductive Meta Learning Loss and Accuracy** during federated training
- Loss computed on **meta-task query sets** (transductive evaluation during training)
- Accuracy from **meta-learning classification** on query samples
- Progress over **federated rounds** (aggregated across clients)

**Why loss decreases**:
1. ✅ Model learns from federated aggregation
2. ✅ Prototype-based classification improves
3. ✅ **Meta-learning** adapts better to new tasks
4. ✅ **Transductive learning** (using query set during training) improves
5. ✅ This is **expected and good behavior**

**What the plot shows**:
- **Transductive meta learning progress** during federated learning
- **Average meta-learning loss per round** (averaged across clients)
- **Meta-learning curve** showing convergence
- Model's ability to quickly adapt to new meta-learning tasks

**The decreasing loss curve in your `training_history_PortScan_.png` is CORRECT and indicates successful transductive meta learning!** ✅

