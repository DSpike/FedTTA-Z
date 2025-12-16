# System Flow Pipeline - Schematic Diagram

## 🎯 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DATA PREPROCESSING PHASE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

    UNSW-NB15 Dataset
         │
         ├─► [Feature Engineering]
         │   ├─ Encoding categorical features
         │   ├─ Handling missing values
         │   └─ Feature scaling (StandardScaler)
         │
         ├─► [Feature Selection]
         │   └─ IG + RF Hybrid Selection (if enabled)
         │
         ├─► [Zero-Day Split]
         │   ├─ Training Set (excludes zero-day attack)
         │   ├─ Validation Set (excludes zero-day attack)
         │   └─ Test Set (includes zero-day attack at 20%)
         │
         └─► [Sequence Creation]
             ├─ sequence_length = 30
             ├─ sequence_stride = 13
             └─ Creates temporal sequences from packets
                 │
                 ├─► X_train_seq, y_train_seq
                 ├─► X_val_seq, y_val_seq
                 └─► X_test_seq, y_test_seq (with multiclass labels for zero-day)


┌─────────────────────────────────────────────────────────────────────────────────┐
│                      FEDERATED LEARNING SETUP                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

    Preprocessed Training Data
         │
         ├─► [Dirichlet Distribution]
         │   ├─ alpha = 1.026 (non-IID data distribution)
         │   └─ Distributes data among N clients (N=8)
         │
         └─► [Client Initialization]
             ├─ Client 1, Client 2, ..., Client 8
             ├─ Each client has local training data
             └─ Each client has a copy of the global model


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FEDERATED TRAINING PHASE (Rounds 1-5)                       │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │                    FEDERATED ROUND                           │
    │                                                              │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
    │  │  Client 1    │  │  Client 2    │  │  Client N    │     │
    │  │              │  │              │  │              │     │
    │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │     │
    │  │ │ Meta-Task│ │  │ │ Meta-Task│ │  │ │ Meta-Task│ │     │
    │  │ │ Creation │ │  │ │ Creation │ │  │ │ Creation │ │     │
    │  │ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │     │
    │  │      │       │  │      │       │  │      │       │     │
    │  │      ▼       │  │      ▼       │  │      ▼       │     │
    │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │     │
    │  │ │  Support │ │  │ │  Support │ │  │ │  Support │ │     │
    │  │ │   Set    │ │  │ │   Set    │ │  │ │   Set    │ │     │
    │  │ │ (169-shot│ │  │ │ (169-shot│ │  │ │ (169-shot│ │     │
    │  │ │ per class│ │  │ │ per class│ │  │ │ per class│ │     │
    │  │ │ Normal+  │ │  │ │ Normal+  │ │  │ │ Normal+  │ │     │
    │  │ │ Attack)  │ │  │ │ Attack)  │ │  │ │ Attack)  │ │     │
    │  │ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │     │
    │  │      │       │  │      │       │  │      │       │     │
    │  │      ▼       │  │      ▼       │  │      ▼       │     │
    │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │     │
    │  │ │  Query   │ │  │ │  Query   │ │  │ │  Query   │ │     │
    │  │ │   Set    │ │  │ │   Set    │ │  │ │   Set    │ │     │
    │  │ │ (~33%    │ │  │ │ (~33%    │ │  │ │ (~33%    │ │     │
    │  │ │ Normal,  │ │  │ │ Normal,  │ │  │ │ Normal,  │ │     │
    │  │ │ ~67%     │ │  │ │ ~67%     │ │  │ │ ~67%     │ │     │
    │  │ │ Attack)  │ │  │ │ Attack)  │ │  │ │ Attack)  │ │     │
    │  │ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │     │
    │  │      │       │  │      │       │  │      │       │     │
    │  │      ▼       │  │      ▼       │  │      ▼       │     │
    │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │     │
    │  │ │Transduct.│ │  │ │Transduct.│ │  │ │Transduct.│ │     │
    │  │ │Meta-Learn│ │  │ │Meta-Learn│ │  │ │Meta-Learn│ │     │
    │  │ │  (TCN +  │ │  │ │  (TCN +  │ │  │ │  (TCN +  │ │     │
    │  │ │Prototype)│ │  │ │Prototype)│ │  │ │Prototype)│ │     │
    │  │ └────┬─────┘ │  │ └────┬─────┘ │  │ └────┬─────┘ │     │
    │  │      │       │  │      │       │  │      │       │     │
    │  └──────┼───────┘  └──────┼───────┘  └──────┼───────┘     │
    │         │                 │                 │              │
    │         └─────────────────┼─────────────────┘              │
    │                           │                                │
    │                           ▼                                │
    │                  ┌─────────────────┐                       │
    │                  │ FedProx         │                       │
    │                  │ Aggregation     │                       │
    │                  │ (with μ=0.001)  │                       │
    │                  └────────┬────────┘                       │
    │                           │                                │
    │                           ▼                                │
    │                  ┌─────────────────┐                       │
    │                  │  Global Model   │                       │
    │                  │    Update       │                       │
    │                  └────────┬────────┘                       │
    │                           │                                │
    │                           ▼                                │
    │              ┌─────────────────────────┐                   │
    │              │  Validation Evaluation  │                   │
    │              │  (X_val, y_val)         │                   │
    │              │  - Overfitting Detection│                   │
    │              │  - No Zero-Day Attacks  │                   │
    │              └─────────────────────────┘                   │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
                         │
                         │ (Repeat for 5 rounds)
                         │
                         ▼


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TTT ADAPTATION PHASE (After Training)                        │
└─────────────────────────────────────────────────────────────────────────────────┘

    Global Model (from Federated Learning)
         │
         ├─► [Model Cloning]
         │   └─ Efficient state_dict-based cloning
         │
         ├─► [Query Set Selection]
         │   └─ Sample from test set (X_test, y_test)
         │       └─ Contains zero-day attacks (20%)
         │
         └─► [TENT + Pseudo-Labels Adaptation]
             │
             ├─► Mini-batch Processing (batch_size=16)
             ├─► Mixed Precision Training (FP16/FP32)
             │
             ├─► Loss Components:
             │   ├─ Entropy Minimization (weight=0.77)
             │   │   └─ Filtered Entropy (max_probs > 0.4)
             │   │
             │   └─ Pseudo-Label Loss (weight=2.06)
             │       ├─ Teacher-Student EMA (decay=0.991)
             │       ├─ Threshold: 0.890 (adaptive)
             │       └─ Temperature: 0.469
             │
             ├─► Optimizer: Adam (lr=6.04e-4)
             ├─► Steps: 332 (adaptive)
             │
             └─► Adapted Model (TTT-Enhanced)
                 │
                 └─► Temperature Scaling (T=1.51)
                     └─ Probability Calibration


┌─────────────────────────────────────────────────────────────────────────────────┐
│                        EVALUATION PHASE                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────┐
    │                   TEST SET EVALUATION                            │
    │                   (X_test, y_test)                               │
    │                   Contains: 19 zero-day + 76 non-zero-day        │
    │                                                                   │
    │  ┌────────────────────────┐    ┌────────────────────────┐      │
    │  │  Base Model Evaluation │    │ TTT Model Evaluation   │      │
    │  │  (No Adaptation)       │    │ (After Adaptation)     │      │
    │  └───────────┬────────────┘    └───────────┬────────────┘      │
    │              │                              │                   │
    │              ▼                              ▼                   │
    │  ┌───────────────────────┐    ┌───────────────────────┐       │
    │  │ Overall Metrics:      │    │ Overall Metrics:      │       │
    │  │ - Accuracy            │    │ - Accuracy            │       │
    │  │ - F1-Score            │    │ - F1-Score            │       │
    │  │ - AUC-PR              │    │ - AUC-PR              │       │
    │  │ - Precision/Recall    │    │ - Precision/Recall    │       │
    │  └───────────┬───────────┘    └───────────┬───────────┘       │
    │              │                              │                   │
    │              ▼                              ▼                   │
    │  ┌───────────────────────┐    ┌───────────────────────┐       │
    │  │ Zero-Day Specific:    │    │ Zero-Day Specific:    │       │
    │  │ - ZDR: 94.74%         │    │ - ZDR: XX.XX%         │       │
    │  │ - Precision: 100%     │    │ - Precision: XX%      │       │
    │  │ - Recall: 94.74%      │    │ - Recall: XX%         │       │
    │  │ - F1-Score: 97.30%    │    │ - F1-Score: XX%       │       │
    │  │ - AUC-PR: 100%        │    │ - AUC-PR: XX%         │       │
    │  └───────────────────────┘    └───────────────────────┘       │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DATA FLOW SUMMARY                                            │
└─────────────────────────────────────────────────────────────────────────────────┘

Dataset (UNSW-NB15)
    │
    ├─► Preprocessing
    │   ├─ Feature Engineering
    │   ├─ Feature Selection
    │   ├─ Zero-Day Split
    │   └─ Sequence Creation
    │       │
    │       ├─► Training Set (sequences)
    │       │   └─ Distributed to Clients (Dirichlet)
    │       │
    │       ├─► Validation Set (sequences)
    │       │   └─ Used during training for overfitting detection
    │       │
    │       └─► Test Set (sequences, 20% zero-day)
    │           └─ Used for final evaluation
    │
    ├─► Federated Learning (5 rounds)
    │   ├─ Each round:
    │   │   ├─ Client local training (meta-learning)
    │   │   ├─ FedProx aggregation
    │   │   └─ Validation evaluation
    │   │
    │   └─ Global Model (after 5 rounds)
    │
    ├─► TTT Adaptation
    │   ├─ Query set from test data
    │   ├─ TENT + Pseudo-Labels
    │   └─ Adapted Model
    │
    └─► Final Evaluation
        ├─ Base Model (on test set)
        └─ TTT Model (on test set)
            ├─ Overall metrics
            └─ Zero-day specific metrics


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KEY COMPONENTS DETAIL                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

1. MODEL ARCHITECTURE:
   ┌─────────────┐
   │  Input      │ (sequence_length=30, features=30)
   │  Sequences  │
   └──────┬──────┘
          │
          ▼
   ┌──────────────────────────────────┐
   │  EfficientMultiScaleTCN          │
   │  - Branch 1: kernel=2            │
   │  - Branch 2: kernel=3            │
   │  - Branch 3: kernel=4            │
   │  (Depthwise Separable Convolutions)│
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │  Feature Projection              │
   │  (embedding_dim=512)             │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │  Prototype-Based Classifier      │
   │  - Support set prototypes        │
   │  - Cosine similarity             │
   │  - Binary classification         │
   └──────────────┬───────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │  Output: Logits                  │
   │  [Normal_prob, Attack_prob]      │
   └──────────────────────────────────┘

2. META-LEARNING TASK STRUCTURE:
   ┌─────────────────────────────┐
   │      Meta-Task (2-way)      │
   │                             │
   │  Support Set:               │
   │  ├─ Normal: 169 samples     │
   │  └─ Attack: 169 samples     │
   │      (from ALL 8 attack     │
   │       types, excluding      │
   │       zero-day)             │
   │                             │
   │  Query Set:                 │
   │  ├─ Normal: ~33%            │
   │  └─ Attack: ~67%            │
   │                             │
   │  Total: 20 tasks per client │
   │         per round           │
   └─────────────────────────────┘

3. TTT ADAPTATION PROCESS:
   Base Model
       │
       ├─ Clone Model
       │
       ├─ Sample Query Set (from test data)
       │   └─ Contains zero-day attacks
       │
       └─ TENT + Pseudo-Labels Loop (332 steps):
           │
           ├─ Forward Pass (mini-batch)
           ├─ Compute Loss:
           │   ├─ Entropy Loss (filtered)
           │   └─ Pseudo-Label Loss
           ├─ Backward Pass (mixed precision)
           ├─ Update Parameters
           └─ EMA Update (teacher model)
               │
               └─ Adapted Model
                   │
                   └─ Temperature Scaling
                       └─ Final Adapted Model


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ZERO-DAY HANDLING                                            │
└─────────────────────────────────────────────────────────────────────────────────┘

Training Set:
    ├─ Zero-Day Attack: EXCLUDED ❌
    └─ Other Attacks: INCLUDED ✅

Validation Set:
    ├─ Zero-Day Attack: EXCLUDED ❌
    └─ Other Attacks: INCLUDED ✅

Test Set:
    ├─ Zero-Day Attack: INCLUDED ✅ (20% of sequences)
    └─ Other Attacks: INCLUDED ✅ (80% of sequences)

Meta-Task Support Set:
    ├─ Zero-Day Attack: EXCLUDED ❌
    ├─ Normal Samples: INCLUDED ✅
    └─ Other Attacks: INCLUDED ✅ (all 8 types)

Meta-Task Query Set (Training):
    ├─ Zero-Day Attack: EXCLUDED ❌
    └─ Normal + Other Attacks: INCLUDED ✅

TTT Query Set (Adaptation):
    ├─ Zero-Day Attack: INCLUDED ✅ (for adaptation)
    └─ Normal + Other Attacks: INCLUDED ✅


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE METRICS OUTPUT                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

1. Training Phase Metrics:
   - Client training loss (per round)
   - Client training accuracy (per round)
   - Validation accuracy (per round)
   - Validation F1-score (per round)
   - Overfitting detection

2. Evaluation Phase Metrics (Base Model):
   - Overall: Accuracy, F1-Score, AUC-PR, Precision, Recall, FAR
   - Zero-Day Specific: ZDR, Precision, Recall, F1-Score, AUC-PR
   - Non-Zero-Day: Accuracy, F1-Score, Precision, Recall

3. Evaluation Phase Metrics (TTT Model):
   - Overall: Accuracy, F1-Score, AUC-PR, Precision, Recall, FAR
   - Zero-Day Specific: ZDR, Precision, Recall, F1-Score, AUC-PR
   - Non-Zero-Day: Accuracy, F1-Score, Precision, Recall

4. Visualizations:
   - Training history plots
   - Confusion matrices (base & TTT)
   - TTT adaptation curves
   - Client performance comparison
   - Performance comparison (base vs TTT)
   - Zero-day performance comparison
   - ROC curves
   - PR curves (PRIMARY metric)


┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KEY CONFIGURATION PARAMETERS                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Federated Learning:
  - num_clients: 8
  - num_rounds: 5
  - local_epochs: 10
  - dirichlet_alpha: 1.026 (non-IID)
  - use_fedprox: True (μ=0.001)

Meta-Learning:
  - n_way: 2 (Normal vs Attack)
  - k_shot: 169 (per class)
  - n_query: 18 (per task)
  - num_meta_tasks: 20 (per client per round)
  - include_all_attack_types_in_support: True

Model Architecture:
  - use_tcn: True
  - tcn_kernel_sizes: (2, 3, 4)
  - hidden_dim: 512
  - embedding_dim: 512
  - sequence_length: 31
  - sequence_stride: 13

TTT Adaptation:
  - ttt_base_steps: 332
  - ttt_batch_size: 16
  - ttt_lr: 6.04e-4
  - ttt_temperature: 1.51
  - entropy_weight: 0.77
  - pseudo_weight: 2.06
  - use_mixed_precision: True

Zero-Day Configuration:
  - zero_day_attack: Backdoor (label 3)
  - zero_day_target_percentage: 20% (in test set)
  - zero_day_excluded_from_training: True
  - zero_day_excluded_from_validation: True
```









