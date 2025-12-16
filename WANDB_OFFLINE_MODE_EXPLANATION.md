# Why Wandb is Set to Offline Mode

## 📋 **Current Configuration**

**Location:** `optimize_hyperparameters.py` lines 57-70

**Code:**
```python
# Use offline mode to prevent hanging on network issues
try:
    wandb.init(
        project="zero-day-detection-optimization",
        name=study_name,
        config={...},
        mode="offline",  # Use offline mode to prevent hanging
        reinit=True
    )
    logger.info("✅ Wandb initialized successfully (offline mode)")
except Exception as e:
    logger.warning(f"⚠️ Wandb initialization failed: {e}. Continuing without wandb logging...")
```

---

## 🎯 **Why Offline Mode?**

### **Primary Reason: Prevent Hanging**

**Problem with Online Mode:**
- Wandb tries to connect to `https://wandb.ai` servers
- Network issues can cause the script to **hang** waiting for connection
- Long optimization runs (10-20 hours) would **block** if network fails
- Script becomes **unresponsive** during network timeouts

**Solution: Offline Mode**
- ✅ **No network connection needed** during optimization
- ✅ **Saves logs locally** (in `wandb/` directory)
- ✅ **Prevents hanging** - optimization continues even if network is down
- ✅ **Can sync later** - upload logs to wandb.ai when convenient

---

## 📊 **What Offline Mode Does**

### **How It Works:**

1. **During Optimization:**
   - Logs all metrics to local files in `wandb/offline-run-*/` directory
   - No network calls during execution
   - Optimization runs smoothly without interruption

2. **After Optimization:**
   - Run `wandb sync wandb/offline-run-*` to upload logs to wandb.ai
   - View results in wandb dashboard
   - All data is preserved locally

### **Local Storage:**
```
wandb/
  └── offline-run-20251202_090708-xxxxx/
      ├── wandb-metadata.json
      ├── wandb-summary.json
      ├── wandb-events.jsonl
      └── logs/
```

---

## ✅ **Benefits of Offline Mode**

### **1. Reliability**
- ✅ Optimization **never hangs** due to network issues
- ✅ Works in **air-gapped environments**
- ✅ No dependency on external services

### **2. Performance**
- ✅ **No network latency** during optimization
- ✅ Faster logging (local file writes vs network uploads)
- ✅ No bandwidth usage during long runs

### **3. Flexibility**
- ✅ Can run optimization **anywhere** (no internet needed)
- ✅ Sync logs **later** when convenient
- ✅ Keep data **private** until ready to share

---

## 🔄 **How to Switch to Online Mode**

### **Option 1: Modify Code (Temporary)**
```python
# In optimize_hyperparameters.py line 67
mode="online",  # Change from "offline" to "online"
```

### **Option 2: Use Environment Variable**
```bash
# Before running optimization
export WANDB_MODE=online
python optimize_hyperparameters.py --n_trials 10
```

### **Option 3: Command-Line (If Supported)**
Some wandb versions support:
```bash
wandb login  # Login first
python optimize_hyperparameters.py --n_trials 10
# Then modify code to use mode="online"
```

---

## 📤 **How to Sync Offline Logs to Wandb.ai**

### **After Optimization Completes:**

```bash
# List offline runs
ls wandb/offline-run-*

# Sync a specific run
wandb sync wandb/offline-run-20251202_090708-xxxxx

# Or sync all offline runs
wandb sync wandb/offline-run-*/
```

### **What Gets Synced:**
- All trial metrics
- Hyperparameters
- System configurations
- Optimization progress
- Best trial results

---

## ⚠️ **Potential Issues with Online Mode**

### **Why Offline is Safer:**

1. **Network Timeouts:**
   - If wandb.ai servers are slow/unavailable
   - Script may hang waiting for response
   - Optimization stalls indefinitely

2. **Authentication Issues:**
   - Requires `wandb login` before running
   - Token expiration can cause failures
   - Missing credentials = script fails

3. **Firewall/Proxy Issues:**
   - Corporate networks may block wandb.ai
   - VPN issues can interrupt connections
   - Script becomes unreliable

4. **Long Runs:**
   - Network connection may drop during 10-20 hour runs
   - Script hangs waiting for reconnect
   - Lost progress if script crashes

---

## 🎯 **Recommendation**

### **Keep Offline Mode For:**
- ✅ **Long optimization runs** (10+ hours)
- ✅ **Unreliable network** environments
- ✅ **Production/experiment runs** (reliability > convenience)
- ✅ **Air-gapped systems**

### **Use Online Mode For:**
- ✅ **Short test runs** (quick feedback)
- ✅ **Reliable network** (always connected)
- ✅ **Real-time monitoring** (want to watch progress live)
- ✅ **Shared experiments** (team wants to see results immediately)

---

## 📊 **Current Behavior**

### **What You See:**
```
✅ Wandb initialized successfully (offline mode)
Run data is saved locally in wandb\offline-run-20251202_090708-xxxxx
```

### **What's Logged:**
- ✅ All metrics (base and TTT)
- ✅ All hyperparameters
- ✅ Trial progress
- ✅ Best trial information

### **Where It's Saved:**
- ✅ Local: `wandb/offline-run-*/` directory
- ✅ Can sync to wandb.ai later

---

## 💡 **Best Practice**

### **Recommended Workflow:**

1. **Run Optimization (Offline):**
   ```bash
   python optimize_hyperparameters.py --n_trials 10
   ```
   - Runs reliably without network
   - All data saved locally

2. **After Completion, Sync:**
   ```bash
   wandb sync wandb/offline-run-*/
   ```
   - Upload logs to wandb.ai
   - View in dashboard
   - Share with collaborators

3. **Keep Offline for Long Runs:**
   - More reliable
   - No risk of hanging
   - Can sync anytime later

---

## ✅ **Summary**

**Why Offline Mode:**
- ✅ **Prevents hanging** on network issues (primary reason)
- ✅ **More reliable** for long runs (10-20 hours)
- ✅ **Works everywhere** (no internet needed)
- ✅ **Safe** - no risk of script blocking

**Trade-offs:**
- ❌ No real-time monitoring (but can sync later)
- ❌ No immediate dashboard access (but data preserved)
- ✅ **All data is saved** - nothing is lost

**Recommendation:** **Keep offline mode** for optimization runs. It's more reliable and you can always sync logs later to view in the wandb dashboard.

---

## 🔧 **Quick Fix (If You Want Online Mode)**

If you really want online mode for real-time monitoring:

1. **Edit `optimize_hyperparameters.py` line 67:**
   ```python
   mode="online",  # Change from "offline"
   ```

2. **Make sure you're logged in:**
   ```bash
   wandb login
   ```

3. **Run optimization:**
   ```bash
   python optimize_hyperparameters.py --n_trials 10
   ```

**Warning:** Ensure you have reliable network connection, otherwise optimization may hang!









