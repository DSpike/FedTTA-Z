# Total Meta-Tasks Used in Experiment

## 📊 **Meta-Tasks Configuration**

### **Per Client Per Round:**

- **`num_meta_tasks = 20`**: Each client creates 20 meta-tasks per federated round

### **Federated Learning Configuration:**

- **`num_clients = 8`**: 8 clients participating
- **`num_rounds = 5`**: 5 federated rounds

---

## 🎯 **Total Meta-Tasks Calculation**

### **Per Round:**

```
Tasks per round = num_clients × num_meta_tasks
                = 8 clients × 20 tasks
                = 160 meta-tasks per round
```

### **Across All Rounds:**

```
Total tasks = num_rounds × tasks per round
           = 5 rounds × 160 tasks
           = 800 meta-tasks total
```

---

## 📋 **Complete Breakdown**

| Level                       | Meta-Tasks | Details                      |
| --------------------------- | ---------- | ---------------------------- |
| **Per Client Per Round**    | **20**     | Each client creates 20 tasks |
| **Per Round (All Clients)** | **160**    | 8 clients × 20 tasks         |
| **Total (All Rounds)**      | **800**    | 5 rounds × 160 tasks         |

---

## 🔍 **Per Client Total**

### **Across All Rounds:**

```
Tasks per client = num_rounds × num_meta_tasks
                 = 5 rounds × 20 tasks
                 = 100 meta-tasks per client
```

---

## 📊 **Summary Table**

| Component                       | Count   | Meta-Tasks |
| ------------------------------- | ------- | ---------- |
| **Per Client Per Round**        | 20      | 20         |
| **Per Round (8 Clients)**       | 160     | 160        |
| **Per Client Total (5 Rounds)** | 100     | 100        |
| **Total Experiment**            | **800** | **800**    |

---

## 🎯 **Simple Answer**

### **In the Experiment:**

- **20 meta-tasks** per client per round
- **160 meta-tasks** per federated round (all clients combined)
- **800 meta-tasks** total (across all 5 rounds)

---

## 📈 **Sample Breakdown**

### **Per Meta-Task:**

- **374 samples** (169 Normal + 169 Attack in support, ~12 Normal + ~24 Attack in query)

### **Total Samples Across Experiment:**

```
Total samples = 800 tasks × 374 samples per task
              = 299,200 sample-task interactions
```

**Note**: This counts interactions, not unique samples (due to sampling with replacement).



