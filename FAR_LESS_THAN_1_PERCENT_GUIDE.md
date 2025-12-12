# Guide: Reducing FAR to Less Than 1%

## ✅ Changes Applied

### 1. **Configuration Updates** (`config.py`)

#### FAR Target Set to 1%
```python
max_far_allowed: float = 0.01  # Changed from 0.20 (20%) to 0.01 (1%)
min_zdr_required: float = 0.75  # Reduced from 0.85 to allow stricter FAR constraint
```

#### Threshold Optimization Strategy
```python
threshold_optimization_strategy: str = 'far_optimized'  # Changed from 'balanced_zdr_far'
```

#### Increased Confidence Rejection Threshold
```python
confidence_rejection_threshold: float = 0.90  # Increased from 0.826 to 0.90
```

### 2. **New FAR-Optimized Strategy** (`main.py`)

Added a new `far_optimized` threshold strategy that:
- Searches thresholds from 0.7 to 0.99 (high thresholds = lower FAR)
- Prioritizes FAR ≤ 1% over ZDR
- Selects the threshold that achieves FAR < 1% while maximizing ZDR
- Falls back gracefully if 1% FAR cannot be achieved

### 3. **Base Model FAR Optimization**

Added FAR-optimized threshold calculation for base model:
- Uses same `max_far_allowed = 0.01` constraint
- Searches high thresholds (0.7-0.99) to minimize FAR
- Applies threshold to base model predictions for FAR calculation

## 📊 Expected Impact

### Before:
- FAR: ~20-50% (depending on threshold strategy)
- ZDR: ~85-95%

### After (Target):
- FAR: **< 1%** ✅
- ZDR: ~70-85% (may decrease slightly due to stricter threshold)

## ⚠️ Trade-offs

1. **ZDR Reduction**: Stricter FAR constraint may reduce ZDR by 5-15%
2. **Higher Threshold**: Thresholds will be higher (0.85-0.95 range)
3. **More Rejections**: Higher confidence threshold (0.90) will reject more uncertain predictions

## 🔧 How It Works

### TTT Model:
1. Uses `far_optimized` strategy by default
2. Searches 500 candidate thresholds from 0.7 to 0.99
3. Selects threshold where FAR ≤ 1%
4. Maximizes ZDR within FAR constraint

### Base Model:
1. Calculates FAR-optimized threshold using same method
2. Applies threshold to attack probabilities
3. Uses threshold-based predictions for FAR calculation

## 🎯 Usage

Simply run the code with the updated `config.py`. The system will automatically:
- Use `far_optimized` strategy for TTT model
- Target FAR ≤ 1%
- Use higher confidence rejection threshold (0.90)
- Optimize thresholds for both base and TTT models

## 📝 Monitoring

Check logs for:
- `✅ Threshold Strategy: FAR-optimized`
- `Results: FAR=X.XXXX (✅ if ≤ 1%)`
- `Selected threshold: X.XXXX` (should be high, e.g., 0.85-0.95)

## 🔄 Reverting Changes

To revert to previous behavior:
```python
# In config.py:
max_far_allowed: float = 0.20  # 20% FAR
threshold_optimization_strategy: str = 'balanced_zdr_far'
confidence_rejection_threshold: float = 0.8261845713819337
```

