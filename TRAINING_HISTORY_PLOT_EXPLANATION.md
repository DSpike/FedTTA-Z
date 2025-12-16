# How training*history_Exploits*.png Evaluation Plot is Generated

## 📊 **Plot Overview**

The `training_history_Exploits_.png` plot shows the federated learning training progress over rounds, displaying both **loss** and **accuracy** evolution.

---

## 🔄 **Complete Data Flow**

### **1. Data Collection During Federated Learning**

During each federated round (in `main.py` lines 6419-6491):

```python
# After each round completes
round_data = {
    'round_number': round_num,
    'client_updates': client_updates,  # List of client update objects
    'avg_loss': round_results.get('avg_loss', 0.0),
    'round_losses': round_losses,
    'round_accuracies': round_accuracies,
    'validation_accuracy': validation_accuracy,
    'validation_loss': validation_loss,
    'training_accuracy': avg_training_accuracy,
    'accuracy_gap': accuracy_gap,
    'overfitting_detected': overfitting_detected
}
system.training_history.append(round_data)
```

**What gets stored:**

- `client_updates`: List of client update objects containing:
  - `training_loss`: Loss from client's local training
  - `validation_accuracy`: Accuracy on client's local data

---

### **2. Data Extraction for Visualization**

In `generate_performance_visualizations()` (main.py lines 2335-2380):

```python
# Extract real training data from federated rounds
epoch_losses = []
epoch_accuracies = []

for round_data in self.training_history:
    if 'client_updates' in round_data and round_data['client_updates']:
        client_updates = round_data['client_updates']

        if isinstance(client_updates, (list, tuple)):
            round_losses = []
            round_accuracies = []

            # Extract metrics from each client
            for client_update in client_updates:
                training_loss = getattr(client_update, 'training_loss', None)
                validation_accuracy = getattr(client_update, 'validation_accuracy', None)

                if training_loss is not None:
                    round_losses.append(training_loss)
                if validation_accuracy is not None:
                    round_accuracies.append(validation_accuracy)

            # Average across all clients for this round
            if round_losses:
                epoch_losses.append(np.mean(round_losses))
            if round_accuracies:
                epoch_accuracies.append(np.mean(round_accuracies))

# Create training history dictionary
training_history = {
    'epoch_losses': epoch_losses,      # One value per round (averaged across clients)
    'epoch_accuracies': epoch_accuracies  # One value per round (averaged across clients)
}
```

**Key Points:**

- **One data point per federated round**
- Each data point is the **average across all clients** for that round
- `epoch_losses`: Average training loss across all clients
- `epoch_accuracies`: Average validation accuracy across all clients

---

### **3. Visualizer Initialization**

The `PerformanceVisualizer` is initialized with the attack name (main.py line ~2295):

```python
# Initialize visualizer with attack name
self.visualizer = PerformanceVisualizer(
    output_dir="performance_plots",
    attack_name=config.zero_day_attack  # e.g., "Exploits"
)
```

**Attack Name Source:**

- Comes from `config.zero_day_attack` (e.g., "Exploits")
- Used in filename generation and plot titles

---

### **4. Plot Generation**

In `visualization/performance_visualization.py` (lines 85-154):

```python
def plot_training_history(self, training_history: Dict[str, List], save: bool = True) -> str:
    """
    Plot training history (loss and accuracy over epochs/rounds)

    Args:
        training_history: Dictionary with 'epoch_losses' and 'epoch_accuracies'
        save: Whether to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract data
    epoch_losses = training_history['epoch_losses']
    epochs = range(1, len(epoch_losses) + 1)  # Round numbers: 1, 2, 3, ...

    # LEFT PLOT: Training Loss
    ax1.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.set_title(f'Federated Training Loss Over Rounds{self._get_title_suffix()}', fontweight='bold')
    # Title includes attack name: "Federated Training Loss Over Rounds (Exploits Attack)"
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Average Loss')
    ax1.grid(True, alpha=0.3)

    # Adaptive scale selection (log if loss ratio > 50x)
    # ... (automatically chooses linear or log scale)

    # Add value labels
    for i, loss in enumerate(epoch_losses):
        ax1.annotate(f'{loss:.4f}', (epochs[i], loss), ...)

    # RIGHT PLOT: Training Accuracy
    ax2.plot(epochs, training_history['epoch_accuracies'], 'g-', linewidth=2, marker='s', markersize=6)
    ax2.set_title('Federated Training Accuracy Over Rounds', fontweight='bold')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Average Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # Add value labels
    for i, acc in enumerate(training_history['epoch_accuracies']):
        ax2.annotate(f'{acc:.3f}', (epochs[i], acc), ...)

    # Save plot
    if save:
        plot_path = os.path.join(self.output_dir, self._get_filename("training_history"))
        # Filename: "training_history_Exploits_.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plt.close()
    return plot_path
```

---

### **5. Filename Generation**

The filename `training_history_Exploits_.png` is generated in `_get_filename()`:

```python
def _get_filename(self, plot_type: str) -> str:
    """
    Generate filename for plot

    Returns:
        filename: "training_history_Exploits_.png"
    """
    # attack_name = "Exploits" (from config.zero_day_attack)
    # timestamp = "" (empty, so no timestamp)
    if self.attack_name:
        return f"{plot_type}_{self.attack_name}_{self.timestamp}.png"
    else:
        return f"{plot_type}_{self.timestamp}.png"
```

**Filename Components:**

- `training_history`: Plot type
- `Exploits`: Zero-day attack name (from `config.zero_day_attack`)
- `_`: Separator (before timestamp)
- `_`: After timestamp (which is empty "")
- `.png`: File extension

**Result:** `training_history_Exploits_.png`

---

## 📊 **What the Plot Shows**

### **Left Panel: Training Loss Over Rounds**

- **X-axis**: Federated learning round number (1, 2, 3, ..., 15)
- **Y-axis**: Average training loss across all clients
- **Data**: One value per round = average of all clients' training losses
- **Scale**: Linear or log (automatically chosen if loss ratio > 50x)
- **Title**: "Federated Training Loss Over Rounds (Exploits Attack)"

### **Right Panel: Training Accuracy Over Rounds**

- **X-axis**: Federated learning round number (1, 2, 3, ..., 15)
- **Y-axis**: Average validation accuracy across all clients
- **Data**: One value per round = average of all clients' validation accuracies
- **Range**: 0 to 1.1 (100% + padding)
- **Title**: "Federated Training Accuracy Over Rounds"

---

## 🔍 **Data Sources**

### **Loss Values:**

- **Source**: `client_update.training_loss` (from each client's local training)
- **Calculation**: `np.mean([client1_loss, client2_loss, ..., clientN_loss])` per round
- **Meaning**: Average training loss across all clients (lower is better)
- **Example**: Round 1 → [6.84, 7.93, 6.84] → Average = 7.20

### **Accuracy Values:**

- **Source**: `client_update.validation_accuracy` (from each client's local validation)
- **Calculation**: `np.mean([client1_acc, client2_acc, ..., clientN_acc])` per round
- **Meaning**: Average validation accuracy across all clients (higher is better)
- **Example**: Round 1 → [0.917, 0.923, 0.945] → Average = 0.928

---

## ⚠️ **Important Notes**

### **1. Client-Level Metrics (Not Global)**

The accuracies shown are **client-level validation accuracies** (on each client's local data), NOT the global validation accuracy on the held-out validation set.

**Why this matters:**

- Clients may show high accuracy on their local data (97-98%)
- But global validation accuracy on held-out set may be lower (90-91%)
- This is expected with non-IID data distribution

**Example:**

- Round 7 Client Average: 97.49% (local validation)
- Round 7 Global Validation: 90.98% (held-out set)
- Gap: 6.51% (expected with non-IID data)

### **2. Averaging Across Clients**

Each round's value is the **average** of all clients:

- Round 7: Client 1 (97.46%), Client 2 (97.34%), Client 3 (97.22%), Client 4 (98.15%), Client 5 (97.72%)
- Plot shows: **Average = 97.58%** for Round 7

### **3. One Point Per Round**

- Each federated round produces **one data point** (averaged across clients)
- With 15 rounds, you get 15 points on the plot
- X-axis represents round numbers, not individual training epochs

### **4. Loss Scale Selection**

- **Linear scale**: Used if loss ratio < 50x (e.g., 6.84 → 3.74)
- **Log scale**: Used if loss ratio > 50x (e.g., 6.84 → 0.001)
- Automatically chosen for better visualization

---

## 📈 **Example Interpretation**

If you see:

- **Round 1**: Loss = 6.84, Accuracy = 0.918
- **Round 2**: Loss = 5.35, Accuracy = 0.923
- **Round 3**: Loss = 3.74, Accuracy = 0.945
- **Round 15**: Loss = 2.50, Accuracy = 0.975

**This means:**

1. ✅ **Loss decreasing**: Model is learning (6.84 → 2.50)
2. ✅ **Accuracy increasing**: Model performance improving (91.8% → 97.5%)
3. ✅ **Converging**: Training is progressing well
4. ✅ **Stable**: No sudden jumps or crashes

---

## 🔧 **Code Locations**

1. **Data Collection**: `main.py` lines 6419-6491 (during federated rounds)
2. **Data Extraction**: `main.py` lines 2335-2380 (in `generate_performance_visualizations()`)
3. **Visualizer Init**: `main.py` line ~2295 (`PerformanceVisualizer(attack_name=...)`)
4. **Plot Generation**: `visualization/performance_visualization.py` lines 85-154 (`plot_training_history()`)
5. **Filename Generation**: `visualization/performance_visualization.py` lines 48-53 (`_get_filename()`)

---

## ✅ **Summary**

The `training_history_Exploits_.png` plot:

### **Data Flow:**

1. ✅ Collected during federated rounds (one data point per round)
2. ✅ Extracted from `training_history` (averaged across clients)
3. ✅ Plotted as loss (left) and accuracy (right) over rounds
4. ✅ Saved with attack name in filename

### **What It Shows:**

- ✅ **Loss**: Average training loss per round (lower is better)
- ✅ **Accuracy**: Average validation accuracy per round (higher is better)
- ✅ **Progress**: How well federated learning is converging

### **Key Characteristics:**

- ✅ Uses **client-level metrics** (not global validation)
- ✅ **Averages across clients** per round
- ✅ **One point per round** (15 rounds = 15 points)
- ✅ Filename includes attack name ("Exploits") from config

**This plot helps you see:**

- ✅ Whether training is converging
- ✅ If loss is decreasing
- ✅ If accuracy is improving
- ✅ Training stability across rounds
