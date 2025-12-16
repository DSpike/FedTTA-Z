# Blockchain Federated Learning for NIDS

## 🚀 Simple Implementation

This project extends your GAN-TCN intrusion detection work to blockchain-based federated learning.

## 📁 Project Structure

```
blockchain_federated_nids/
├── blockchain/
│   └── smart_contracts/
│       └── federated_learning_contract.py
├── federated_learning/
│   └── clients/
│       └── federated_client.py
├── config/
│   └── blockchain_config.py
├── main.py
├── run_demo.py
└── requirements_simple.txt
```

## 🚀 Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements_simple.txt
   ```

2. **Test the system:**

   ```bash
   python run_demo.py
   ```

3. **Run federated learning:**
   ```bash
   python main.py
   ```

## 🎯 Key Features

- **Simple Smart Contract**: Model aggregation and incentive distribution
- **Federated Clients**: Local GAN-TCN training
- **Quality-Based Rewards**: Incentives based on model performance
- **Privacy Preserved**: No raw data sharing

## 📊 Expected Results

- Multiple clients train locally
- Models aggregated via blockchain
- Quality-based incentive distribution
- Improved global model performance

## 🔧 Configuration

Edit `config/blockchain_config.py` to adjust:

- Number of clients
- Training rounds
- Learning parameters
- Incentive structur
l