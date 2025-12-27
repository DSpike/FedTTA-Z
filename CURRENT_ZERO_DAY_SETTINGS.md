# Current Zero-Day Attack Settings

## 📊 **Current Configuration**

- **Zero-Day Attack**: `DoS`
- **Attack Grouping**: `False` (fine-grained attack types)
- **Dataset**: `UNSW-NB15`
- **Zero-Day Attack Label**: `4` (should be, but showing as 0 - needs verification)

## 🎯 **Available UNSW Attack Types**

Since `use_category_grouping = False`, you must use **specific attack names**:

| Attack Name | Label | Description |
|------------|-------|-------------|
| Normal | 0 | Normal traffic |
| Fuzzers | 1 | Fuzzing attacks |
| Analysis | 2 | Analysis attacks |
| Backdoor | 3 | Backdoor attacks |
| **DoS** | **4** | **Denial of Service (CURRENT)** |
| Exploits | 5 | Exploit attacks |
| Generic | 6 | Generic attacks |
| Reconnaissance | 7 | Reconnaissance attacks |
| Shellcode | 8 | Shellcode attacks |
| Worms | 9 | Worm attacks |

## ⚙️ **To Change Zero-Day Attack**

Edit `config_loader.py` line 72:
```python
'zero_day_attack': "DoS",  # Change to desired attack name
```

## 🔄 **To Enable/Disable Grouping**

Edit `config_loader.py` line 73:
```python
'use_category_grouping': False,  # Set to True to enable grouping
```

**Note**: If grouping is enabled, you must use **category names** instead of specific attack names.



