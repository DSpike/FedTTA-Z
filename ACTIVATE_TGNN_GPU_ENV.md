# How to Activate Tgnn_gpu Environment

## 🔍 **Problem Identified**

You have **TWO separate Python environments**:

### ❌ **Currently Active (WRONG):**
- **Environment**: System Python 3.10
- **Location**: `C:\Users\Dspike\AppData\Local\Programs\Python\Python310\python.exe`
- **PyTorch**: 2.7.1+**cpu** (CPU-only, no CUDA)
- **Performance**: 4+ minutes per run

### ✅ **Correct Environment (NOT ACTIVE):**
- **Environment**: `Tgnn_gpu` virtual environment
- **Location**: `C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn_gpu`
- **PyTorch**: 2.5.1+**cu121** (CUDA 12.1 support)
- **GPU**: NVIDIA GeForce RTX 4070 Ti SUPER
- **Expected Performance**: ~30-60 seconds per run (10-20x faster!)

---

## ✅ **Solution: Activate Tgnn_gpu Environment**

### **Option 1: From Tgnn Project Folder (Recommended)**

```cmd
cd C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn
..\Tgnn_gpu\Scripts\activate
```

You should see:
```
(Tgnn_gpu) C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn>
```

### **Option 2: Navigate to Tgnn_gpu First**

```cmd
cd C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn_gpu
Scripts\activate
cd ..\Tgnn
```

### **Option 3: PowerShell**

```powershell
cd C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn
..\Tgnn_gpu\Scripts\Activate.ps1
```

**If you get execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
..\Tgnn_gpu\Scripts\Activate.ps1
```

---

## 🎯 **Verify Activation**

### **1. Check Environment Prefix:**
Your command prompt should show:
```
(Tgnn_gpu) C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn>
```

### **2. Check Python Location:**
```cmd
where python
```

**Expected (first line):**
```
C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn_gpu\Scripts\python.exe
```

### **3. Check GPU Availability:**
```cmd
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('PyTorch Version:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Expected Output:**
```
CUDA Available: True
PyTorch Version: 2.5.1+cu121
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
```

---

## 🚀 **Run Your Code**

After activation:
```cmd
python main.py
```

**You should see:**
```
Device: cuda
Centralized Coordinator model moved to device: cuda
✅ Using GPU: NVIDIA GeForce RTX 4070 Ti SUPER
```

**Expected runtime: ~30-60 seconds** (instead of 4+ minutes!)

---

## 🔧 **VSCode Integration (Optional)**

Set VSCode to use the Tgnn_gpu Python automatically:

1. Press `Ctrl+Shift+P`
2. Type: "Python: Select Interpreter"
3. Select: `C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn_gpu\Scripts\python.exe`

Or add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "C:/Users/Dspike/Documents/PhD/TNN/exp1/Tgnn_gpu/Scripts/python.exe"
}
```

---

## 📊 **Performance Comparison**

| Environment | PyTorch | Device | Runtime | Speedup |
|-------------|---------|--------|---------|---------|
| **Current (System Python)** | 2.7.1+cpu | CPU | 4m 17s | 1x |
| **Tgnn_gpu (Correct)** | 2.5.1+cu121 | RTX 4070 Ti SUPER | ~30-60s | **10-20x** ⚡ |

---

## 🎓 **Why This Happens**

When you run `python main.py` from the Tgnn folder **without activating** the virtual environment:
- It uses the system Python (3.10)
- Which has CPU-only PyTorch installed
- Your GPU sits idle while CPU struggles

When you **activate** Tgnn_gpu first:
- It uses the virtual environment Python
- Which has CUDA-enabled PyTorch
- Your RTX 4070 Ti SUPER accelerates training by 10-20x!

---

## 📝 **Quick Commands Summary**

```cmd
# 1. Navigate to project
cd C:\Users\Dspike\Documents\PhD\TNN\exp1\Tgnn

# 2. Activate GPU environment
..\Tgnn_gpu\Scripts\activate

# 3. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 4. Run code (should take ~30-60 seconds now!)
python main.py

# 5. Deactivate when done
deactivate
```




