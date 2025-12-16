# How to Switch to UNSW-NB15 and Run main.py

## 🎯 **Quick Steps to Switch and Run**

---

## **Method 1: Create Branch First (Recommended - Preserves Your Work)**

### **Step 1: Save Your Current Work**

```bash
# Commit any uncommitted changes to master first
git add config.py main.py
git commit -m "Save: CICIDS2017 work before switching to UNSW"

# OR create a backup branch immediately
git checkout -b cicids2017-backup
git add -A
git commit -m "Backup: CICIDS2017 configuration"
git checkout master
```

### **Step 2: Create UNSW Branch from GitHub Version**

```bash
# Create branch from commit e85e10d (has UNSW configuration)
git checkout -b unsw-nb15-version e85e10d

# Verify you're on the new branch
git branch
# Should show: * unsw-nb15-version
```

### **Step 3: Verify UNSW Configuration**

```bash
# Check config shows UNSW
python -c "from config import SystemConfig; c = SystemConfig(); print(f'Dataset: {c.data_path}'); print(f'Zero-day: {c.zero_day_attack}')"
```

**Expected output:**
```
Dataset: UNSW_NB15_training-set.csv
Zero-day: Analysis
```

### **Step 4: Verify Dataset Files Exist**

```bash
# Check if UNSW files exist
ls UNSW_NB15_training-set.csv
ls UNSW_NB15_testing-set.csv
```

**⚠️ If files don't exist:** You need to get the UNSW-NB15 dataset files first!

### **Step 5: Run main.py**

```bash
# Make sure you're on UNSW branch
git checkout unsw-nb15-version

# Run the main script
python main.py
```

---

## **Method 2: Quick Switch (Direct Checkout - No Branch Creation)**

If you just want to test UNSW quickly without creating branches:

### **Step 1: Stash Current Changes**

```bash
# Save current work temporarily
git stash save "Temporary: CICIDS2017 work"

# Verify stash was created
git stash list
```

### **Step 2: Checkout UNSW Commit Directly**

```bash
# Go to the UNSW version commit (detached HEAD state)
git checkout e85e10d

# Verify configuration
python -c "from config import SystemConfig; c = SystemConfig(); print(f'{c.data_path}, {c.zero_day_attack}')"
```

### **Step 3: Run main.py**

```bash
python main.py
```

### **Step 4: Switch Back to Your Work**

```bash
# Go back to master
git checkout master

# Restore your stashed changes
git stash pop
```

**⚠️ Note:** In detached HEAD state, any commits won't be on a branch. Use Method 1 for permanent work.

---

## **Method 3: Using Existing switch_dataset.py Script**

If you have the script working:

```bash
# Make sure you're on master
git checkout master

# Use the script to switch
python switch_dataset.py unsw

# Verify switch worked
python -c "from config import SystemConfig; c = SystemConfig(); print(f'{c.data_path}')"

# Run main.py
python main.py
```

---

## 🔄 **Switching Back to CICIDS2017**

### **If You Used Method 1 (Branch):**
```bash
git checkout cicids2017-backup
# OR
git checkout master
```

### **If You Used Method 2 (Detached HEAD):**
```bash
git checkout master
git stash pop
```

### **If You Used Method 3 (Script):**
```bash
python switch_dataset.py cicids
```

---

## ✅ **Complete Workflow Example**

```bash
# === SETUP (One-time) ===

# 1. Create backup branch for CICIDS work
git checkout master
git checkout -b cicids2017-backup
git commit -am "Backup: CICIDS2017 configuration"
git checkout master

# 2. Create UNSW branch
git checkout -b unsw-nb15-version e85e10d

# === DAILY WORK ===

# Switch to UNSW and run
git checkout unsw-nb15-version
python main.py

# Switch back to CICIDS and run
git checkout cicids2017-backup
python main.py
```

---

## 🔍 **Troubleshooting**

### **Error: "fatal: reference is not a tree: e85e10d"**

**Solution:**
```bash
# Fetch from remote first
git fetch origin

# Try again
git checkout -b unsw-nb15-version e85e10d
```

### **Error: "FileNotFoundError: UNSW_NB15_training-set.csv"**

**Solution:**
- Download/obtain the UNSW-NB15 dataset files
- Place them in your working directory
- Or continue using CICIDS2017 dataset

### **Error: Import errors after switching**

**Solution:**
```bash
# Clear Python cache
Remove-Item -Recurse -Force __pycache__
Remove-Item -Recurse -Force models/__pycache__
Remove-Item -Recurse -Force coordinators/__pycache__

# Try again
python main.py
```

---

## 📋 **Quick Reference**

```bash
# Switch to UNSW
git checkout unsw-nb15-version
python main.py

# Switch to CICIDS
git checkout cicids2017-backup
python main.py

# Check current branch
git branch

# Check current dataset
python -c "from config import SystemConfig; c = SystemConfig(); print(c.data_path)"
```









