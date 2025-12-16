# Safe GitHub Restore Strategy

## 🎯 **Goal: Restore to GitHub Version (UNSW-NB15) While Keeping Your CICIDS2017 Work Safe**

---

## ✅ **Recommended Approach: Git Branch Strategy (SAFEST)**

This approach creates separate branches for each dataset, allowing you to switch between them easily.

### **Step 1: Create a Backup Branch for Your Current Work**

```bash
# Save your current CICIDS2017 work to a new branch
git checkout -b cicids2017-current
git add -A
git commit -m "Backup: Current CICIDS2017 configuration with transductive learning fixes"
```

**This creates a permanent backup of your current work!** ✅

---

### **Step 2: Create a Branch from GitHub Version (UNSW-NB15)**

```bash
# Go back to master (your current commits are safe on cicids2017-current branch)
git checkout master

# Create a new branch from the remote GitHub version (before your CICIDS commits)
# This gives you the UNSW-NB15 version
git checkout -b unsw-nb15-github e85e10d
# OR if you want the absolute latest from GitHub:
git checkout -b unsw-nb15-github origin/master~2  # Go back 2 commits before your recent ones
```

**This creates a separate branch with UNSW-NB15 configuration!** ✅

---

### **Step 3: Switch Between Datasets**

```bash
# To work with CICIDS2017 (your current work):
git checkout cicids2017-current

# To work with UNSW-NB15 (GitHub version):
git checkout unsw-nb15-github
```

**You can switch anytime without losing work!** ✅

---

## 🔧 **Alternative Approach: Use switch_dataset.py Script**

You already have a `switch_dataset.py` script! This might be safer than Git operations.

### **Step 1: Check if switch_dataset.py works**

```bash
# Test the script (doesn't change anything, just shows what it would do)
python switch_dataset.py --help
python switch_dataset.py unsw  # Check what it does (don't run yet)
```

### **Step 2: Backup config.py First**

```bash
# Create backup of current config
cp config.py config_cicids2017_backup.py

# Create backup of main.py preprocessor section
# (manually save the CICIDS preprocessor lines)
```

### **Step 3: Use the Script (If Available)**

If `switch_dataset.py` works, it should handle:
- ✅ Switching preprocessor in `main.py`
- ✅ Switching attack_types in `config.py`
- ✅ Switching data paths
- ✅ Switching zero-day attack

---

## 🛡️ **Safest Manual Approach (Step-by-Step)**

If you want to manually restore while keeping backups:

### **Step 1: Create Backups**

```bash
# Backup current config
cp config.py config_cicids2017_backup.py

# Backup main.py (especially preprocessor section)
cp main.py main_cicids2017_backup.py

# Or create a backup branch
git checkout -b backup-cicids2017
git add config.py main.py
git commit -m "Backup: CICIDS2017 configuration"
git checkout master
```

---

### **Step 2: Restore config.py from GitHub**

```bash
# Restore config.py to GitHub version
git checkout HEAD~2 -- config.py  # Go back 2 commits before your recent changes
# OR
git checkout origin/master~2 -- config.py  # From remote (if you know the commit hash)

# Review what changed
git diff config_cicids2017_backup.py config.py
```

**Check:** Verify it has UNSW-NB15 settings (data_path, attack_types, etc.)

---

### **Step 3: Restore preprocessor in main.py**

**Option A: Manual Edit (Safer)**
```python
# In main.py, lines 450-460:
# Comment out CICIDS:
# from blockchain_federated_cicids_preprocessor import CICIDSPreprocessor
# self.preprocessor = CICIDSPreprocessor(...)

# Uncomment UNSW:
from preprocessing.blockchain_federated_unsw_preprocessor import UNSWPreprocessor
self.preprocessor = UNSWPreprocessor(
    data_path=self.config.data_path,
    test_path=self.config.test_path
)
```

**Option B: Restore from GitHub**
```bash
# Restore main.py preprocessor section from GitHub
git show HEAD~2:main.py | Select-String -Pattern "preprocessor" -Context 10
# Then manually copy the UNSW preprocessor section
```

---

### **Step 4: Verify Dataset Files Exist**

```bash
# Check if UNSW-NB15 files exist
ls UNSW_NB15_training-set.csv
ls UNSW_NB15_testing-set.csv
```

**If files don't exist:** Code will fail at runtime. You need to:
- Either get the UNSW-NB15 dataset files
- OR keep using CICIDS2017

---

### **Step 5: Test the Restore**

```bash
# Quick syntax check
python -c "from config import SystemConfig; c = SystemConfig(); print(f'Dataset: {c.data_path}, Zero-day: {c.zero_day_attack}')"
```

**Expected output:**
```
Dataset: UNSW_NB15_training-set.csv, Zero-day: Analysis
```

---

## 🔄 **How to Switch Back to CICIDS2017**

### **If You Used Branch Strategy:**
```bash
git checkout cicids2017-current
```

### **If You Used Manual Restore:**
```bash
# Restore from backup
cp config_cicids2017_backup.py config.py
cp main_cicids2017_backup.py main.py

# OR restore from Git
git checkout backup-cicids2017 -- config.py main.py
```

---

## ⚠️ **Important Warnings**

### **1. Saved Test Sets**
- CICIDS2017 saved test sets won't work with UNSW-NB15
- System will regenerate test sets (this is OK)
- Location: `saved_test_sets/test_set_*.pkl`

### **2. Saved Models**
- Models trained on CICIDS2017 may have wrong input dimension for UNSW
- You'll need to retrain models

### **3. Hyperparameters**
- GitHub version has hyperparameters optimized for UNSW-NB15
- Performance may differ when using UNSW data

### **4. Zero-Day Attack**
- GitHub uses "Analysis" (UNSW-NB15)
- Your current uses "PortScan" (CICIDS2017)
- Different attack = different experiment!

---

## 📋 **Pre-Restore Checklist**

Before restoring, verify:

- [ ] UNSW-NB15 dataset files exist (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`)
- [ ] You've created a backup branch or backup files
- [ ] You understand you'll need to retrain models
- [ ] You're OK with different zero-day attack ("Analysis" instead of "PortScan")
- [ ] You know how to switch back to CICIDS2017

---

## 🎯 **My Recommendation**

**Option 1: Use Git Branches (BEST)** ⭐
- ✅ Safest - no risk of losing work
- ✅ Easy switching
- ✅ Clean separation

**Steps:**
```bash
# 1. Backup current work
git checkout -b cicids2017-current
git commit -am "Backup: CICIDS2017 work"

# 2. Find the GitHub commit hash (before your CICIDS changes)
git log --oneline | Select-String "UNSW\|Analysis"

# 3. Create UNSW branch from that commit
git checkout -b unsw-nb15-github <commit-hash>

# 4. Switch between them
git checkout cicids2017-current  # For CICIDS
git checkout unsw-nb15-github    # For UNSW
```

**Option 2: Manual Restore with Backups** 
- ✅ More control
- ⚠️ More steps
- ⚠️ Need to be careful

---

## 🔍 **How to Find the Right GitHub Commit**

```bash
# See all commits
git log --oneline

# Look for commits before your recent changes
# Your recent commits:
# 63d8dae - Update documentation
# e11ec1c - Implement true transductive learning
# <earlier> - GitHub version with UNSW

# Find the commit with UNSW configuration
git log --all --grep="UNSW\|Analysis" --oneline
# OR check commit messages around e85e10d (from push output)
```

---

## ✅ **Final Recommendation**

**Use Git Branch Strategy:**
1. Create `cicids2017-current` branch (backup your work)
2. Create `unsw-nb15-github` branch from GitHub commit
3. Work on whichever branch you need
4. Switch freely between them

**This is the SAFEST approach - no risk of losing anything!** 🛡️









