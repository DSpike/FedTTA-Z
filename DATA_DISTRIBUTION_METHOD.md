# Data Distribution Method in Current System

## Current Implementation

**File**: `coordinators/simple_fedavg_coordinator.py`  
**Method**: `distribute_data()` (lines 226-238)  
**Type**: **Simple Sequential Splitting (IID-like)**

### Current Distribution:

- **Method**: Sequential splitting (no randomization, no Dirichlet)
- **Behavior**: Data split into chunks sequentially
- **Result**: Nearly IID distribution (similar class proportions across clients)

```python
chunk_size = len(train_data) // self.num_clients
for i, client in enumerate(self.clients):
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size if i < self.num_clients - 1 else len(train_data)
    client_data = train_data[start_idx:end_idx]
```

## Non-IID Distribution (Dirichlet) - Available in Blockchain Coordinator

**File**: `coordinators/blockchain_fedavg_coordinator.py`  
**Method**: `distribute_data_with_dirichlet()` (lines 1042-1082)  
**Default Alpha (α)**: **1.0**

### Dirichlet Alpha (α) Parameter Values:

| **α Value**  | **Heterogeneity Level** | **Description**      | **Use Case**              |
| ------------ | ----------------------- | -------------------- | ------------------------- |
| **α = 0.1**  | Very High               | Extreme non-IID      | Stress testing            |
| **α = 0.5**  | High                    | High heterogeneity   | Realistic scenarios       |
| **α = 1.0**  | **Moderate**            | **Balanced non-IID** | **RECOMMENDED (Default)** |
| **α = 5.0**  | Low                     | Low heterogeneity    | Near-IID baseline         |
| **α = 10.0** | Very Low                | Near-IID             | IID comparison            |

### Current Default (if using blockchain coordinator):

```python
alpha: float = 1.0  # Moderate heterogeneity (balanced non-IID) - RECOMMENDED
```

---

## Summary

**Current System (`simple_fedavg_coordinator.py`):**

- ❌ **NO non-IID distribution constant** - Uses simple sequential splitting
- Distribution: IID-like (sequential chunks)
- **No Dirichlet alpha parameter**

**Alternative (`blockchain_fedavg_coordinator.py`):**

- ✅ Has Dirichlet distribution with **α = 1.0** (default)
- Creates realistic non-IID data distribution
- **Not currently used** (system uses simple coordinator)

---

## To Enable Non-IID Distribution:

If you want to use Dirichlet distribution (non-IID), you would need to:

1. Add `distribute_data_with_dirichlet()` method to `simple_fedavg_coordinator.py`
2. Add `dirichlet_alpha` parameter to `config.py`
3. Update `distribute_data()` to use Dirichlet instead of simple splitting

