"""Extract best trial from Optuna study database for CICIDS2017"""
import optuna
import json

# Try to load the study
try:
    # Try different study names
    study_names = [
        "cicids_zero_day_detection_optimization",
        "CICIDS2017_optimization",
        "cicids_optimization"
    ]
    
    study = None
    for name in study_names:
        try:
            study = optuna.load_study(study_name=name, storage="sqlite:///optuna_study.db")
            print(f"✅ Found study: {name}")
            break
        except:
            continue
    
    if study is None:
        # Try to list all studies
        print("Available studies in database:")
        import sqlite3
        try:
            conn = sqlite3.connect('optuna_study.db')
            cursor = conn.cursor()
            # Check if studies table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='studies'")
            if cursor.fetchone():
                cursor.execute("SELECT study_name FROM studies")
                studies = cursor.fetchall()
                if studies:
                    for s in studies:
                        print(f"  - {s[0]}")
                    study_name = studies[0][0]
                    study = optuna.load_study(study_name=study_name, storage="sqlite:///optuna_study.db")
                    print(f"✅ Using study: {study_name}")
                else:
                    print("  No studies found in database")
            else:
                print("  Studies table does not exist - database is empty or corrupted")
            conn.close()
        except Exception as e:
            print(f"  Error reading database: {e}")
    
    if study:
        best_trial = study.best_trial
        print(f"\n{'='*80}")
        print(f"Best Trial: {best_trial.number}")
        print(f"Best Value: {best_trial.value}")
        print(f"{'='*80}\n")
        
        print("Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        print(f"\nBest Trial Metrics:")
        for key, value in best_trial.user_attrs.items():
            print(f"  {key}: {value}")
        
        # Save to JSON
        output = {
            "dataset": "CICIDS2017",
            "zero_day_attack": "PortScan",
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_user_attrs": best_trial.user_attrs
        }
        
        with open("best_hyperparameters_cicids.json", "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Saved to best_hyperparameters_cicids.json")
        
        # Print TCN-specific values
        print(f"\n{'='*80}")
        print("TCN Configuration:")
        print(f"  tcn_kernel_sizes: ({best_trial.params.get('tcn_kernel_size_1', 'N/A')}, "
              f"{best_trial.params.get('tcn_kernel_size_2', 'N/A')}, "
              f"{best_trial.params.get('tcn_kernel_size_3', 'N/A')})")
        print(f"  sequence_length: {best_trial.params.get('sequence_length', 'N/A')}")
        print(f"  sequence_stride: {best_trial.params.get('sequence_stride', 'N/A')}")
        print(f"{'='*80}")
    else:
        print("❌ Could not find any study in database")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

