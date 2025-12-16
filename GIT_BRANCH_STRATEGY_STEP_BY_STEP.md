# Git Branch Strategy - Step-by-Step Guide

## 🎯 **Goal: Create Separate Branches for CICIDS2017 and UNSW-NB15**

This allows you to switch between datasets without losing any work.

---

## 📋 **Prerequisites**

- ✅ Git repository initialized
- ✅ Your current work is on `master` branch
- ✅ You have uncommitted changes (if any)

---

## 🔍 **Step 1: Check Current Status**

First, let's see what we're working with:

```bash
# Check current branch
git branch

# Check commit history (last 10 commits)
git log --oneline -10

# Check for uncommitted changes
git status
```

**Expected:** You should be on `master` branch with recent commits including your transductive learning fixes.

---

## 💾 **Step 2: Save Any Uncommitted Changes (If Any)**

If you have uncommitted changes that you want to keep:

```bash
# Check what files are modified
git status

# If you have important uncommitted changes, commit them first
git add <files-to-commit>
git commit -m "WIP: Current changes before branch creation"

# OR stash them temporarily
git stash save "Temporary stash before branching"
```

**Note:** After creating branches, you can apply stashed changes with `git stash pop`.

---

## 🌿 **Step 3: Create Backup Branch for Your Current CICIDS2017 Work**

This branch will preserve all your current CICIDS2017 work:

```bash
# Make sure you're on master
git checkout master

# Create a new branch from current state (your CICIDS2017 work)
git checkout -b cicids2017-backup

# Verify you're on the new branch
git branch

# Commit any uncommitted changes to this branch (if needed)
git add -A
git commit -m "Backup: CICIDS2017 configuration with transductive learning fixes and optimizations"
```

**Result:** Your current CICIDS2017 work is now safely stored on `cicids2017-backup` branch! ✅

---

## 🔍 **Step 4: Find the GitHub Version Commit (UNSW-NB15)**

We need to find the commit that has UNSW-NB15 configuration (before your CICIDS changes):

```bash
# Go back to master
git checkout master

# Look at commit history to find where UNSW-NB15 was
git log --oneline --all -20

# Look for commits with UNSW-related messages or check remote
git log --oneline origin/master -10

# Search for UNSW or Analysis in commit messages
git log --all --grep="UNSW\|Analysis\|unsw" --oneline
```

**What to look for:**
- Commits before your recent transductive learning commits
- Commit messages mentioning UNSW or Analysis
- The commit hash (e.g., `e85e10d`) from the push output earlier

**Alternative:** Check the remote repository to see what commit hash has UNSW configuration:

```bash
# See what's on remote
git log origin/master --oneline -10

# Compare with your local commits
git log master --oneline -5
```

---

## 🌿 **Step 5: Create UNSW-NB15 Branch from GitHub Version**

Once you find the commit hash with UNSW configuration:

```bash
# Make sure you're on master
git checkout master

# Create new branch from the GitHub version commit
# Replace <commit-hash> with the actual hash you found
git checkout -b unsw-nb15-version <commit-hash>

# Example (if commit hash is e85e10d):
# git checkout -b unsw-nb15-version e85e10d

# Verify you're on the new branch
git branch

# Verify the config has UNSW settings
python -c "from config import SystemConfig; c = SystemConfig(); print(f'Dataset: {c.data_path}')"
```

**Expected output:** `Dataset: UNSW_NB15_training-set.csv`

**If you don't know the exact commit hash**, you can:
1. Check your push output - it showed `e85e10d` as the old master
2. Or go back N commits: `git checkout -b unsw-nb15-version HEAD~3` (adjust number)

---

## ✅ **Step 6: Verify Both Branches Are Created**

```bash
# List all branches
git branch -a

# You should see:
# * unsw-nb15-version  (current branch, if you just created it)
#   cicids2017-backup
#   master
```

---

## 🔄 **Step 7: Switching Between Branches**

Now you can switch between datasets easily:

### **To work with CICIDS2017:**
```bash
git checkout cicids2017-backup
# Verify
python -c "from config import SystemConfig; c = SystemConfig(); print(f'Dataset: {c.data_path}, Zero-day: {c.zero_day_attack}')"
# Should show: Dataset: CICIDS2017_train.csv, Zero-day: PortScan
```

### **To work with UNSW-NB15:**
```bash
git checkout unsw-nb15-version
# Verify
python -c "from config import SystemConfig; c = SystemConfig(); print(f'Dataset: {c.data_path}, Zero-day: {c.zero_day_attack}')"
# Should show: Dataset: UNSW_NB15_training-set.csv, Zero-day: Analysis
```

### **To work on master (if needed):**
```bash
git checkout master
```

---

## 🛡️ **Step 8: Verify Dataset Files Exist**

Before running code, make sure the dataset files exist:

### **For CICIDS2017 branch:**
```bash
git checkout cicids2017-backup
ls CICIDS2017_train.csv
ls CICIDS2017_test.csv
```

### **For UNSW-NB15 branch:**
```bash
git checkout unsw-nb15-version
ls UNSW_NB15_training-set.csv
ls UNSW_NB15_testing-set.csv
```

**⚠️ Warning:** If files don't exist, the code will fail at runtime!

---

## 📝 **Step 9: Update Master Branch (Optional)**

You can decide which branch becomes your default `master`:

### **Option A: Keep CICIDS2017 on master (Recommended)**
```bash
git checkout master
git merge cicids2017-backup
# OR
git checkout cicids2017-backup
git branch -m master old-master  # Rename old master
git checkout -b master           # Create new master from cicids2017-backup
```

### **Option B: Keep UNSW-NB15 on master**
```bash
git checkout master
git merge unsw-nb15-version
```

### **Option C: Keep both separate (Best)**
- Use `cicids2017-backup` for CICIDS work
- Use `unsw-nb15-version` for UNSW work
- Keep `master` as is (or update to your preference)

---

## 🔄 **Step 10: Daily Workflow**

### **Working with CICIDS2017:**
```bash
# Switch to CICIDS branch
git checkout cicids2017-backup

# Make changes, run experiments
python main.py

# Commit changes
git add -A
git commit -m "Update: CICIDS2017 experiment results"
```

### **Working with UNSW-NB15:**
```bash
# Switch to UNSW branch
git checkout unsw-nb15-version

# Make changes, run experiments
python main.py

# Commit changes
git add -A
git commit -m "Update: UNSW-NB15 experiment results"
```

---

## 🔍 **Step 11: Verify Branch Isolation**

Make sure branches are properly isolated:

```bash
# Check config.py on CICIDS branch
git show cicids2017-backup:config.py | Select-String "data_path"
# Should show: data_path: str = "CICIDS2017_train.csv"

# Check config.py on UNSW branch
git show unsw-nb15-version:config.py | Select-String "data_path"
# Should show: data_path: str = "UNSW_NB15_training-set.csv"
```

---

## ⚠️ **Troubleshooting**

### **Problem: Can't find the UNSW commit hash**

**Solution:**
```bash
# Try going back a few commits from your recent ones
git log --oneline -10
# Find the commit before your transductive learning commits
# Create branch from that commit

# OR check remote
git fetch origin
git log origin/master --oneline -10
```

### **Problem: Branch has wrong configuration**

**Solution:**
```bash
# Delete the wrong branch
git branch -D unsw-nb15-version

# Try a different commit hash
git checkout -b unsw-nb15-version <different-commit-hash>
```

### **Problem: Uncommitted changes when switching branches**

**Solution:**
```bash
# Commit them first
git add -A
git commit -m "WIP: Current changes"

# OR stash them
git stash
# Switch branch
git checkout other-branch
# If needed, apply stash later
git stash pop
```

---

## ✅ **Quick Reference Commands**

```bash
# Create CICIDS backup branch
git checkout master
git checkout -b cicids2017-backup
git commit -am "Backup: CICIDS2017 work"

# Create UNSW branch from GitHub version
git checkout master
git checkout -b unsw-nb15-version <commit-hash>

# Switch branches
git checkout cicids2017-backup   # CICIDS2017
git checkout unsw-nb15-version   # UNSW-NB15
git checkout master              # Master branch

# List branches
git branch

# Verify configuration
python -c "from config import SystemConfig; c = SystemConfig(); print(f'{c.data_path}, {c.zero_day_attack}')"
```

---

## 🎯 **Summary**

1. ✅ **Create `cicids2017-backup` branch** - Saves your current work
2. ✅ **Find GitHub commit** - Identify UNSW-NB15 version
3. ✅ **Create `unsw-nb15-version` branch** - From GitHub commit
4. ✅ **Switch between branches** - Use `git checkout` anytime
5. ✅ **Work on each independently** - No conflicts, no data loss

**Benefits:**
- 🛡️ Zero risk of losing work
- 🔄 Easy switching between datasets
- 📦 Clean separation of experiments
- 🔙 Can always go back









