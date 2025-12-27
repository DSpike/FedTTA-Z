#!/usr/bin/env python3
"""Check ZDR results from latest run"""
import json
import glob
import os

# Find latest performance metrics file
files = sorted(glob.glob('performance_plots/performance_metrics*.json'), 
               key=lambda x: os.path.getmtime(x), reverse=True)

if files:
    f = files[0]
    print(f"📂 Loading: {f}")
    with open(f, 'r') as file:
        data = json.load(file)
    
    ttt_zdr = data.get('ttt_zero_day', {}).get('zero_day_detection_rate', 'N/A')
    base_zdr = data.get('base_zero_day', {}).get('zero_day_detection_rate', 'N/A')
    ttt_threshold = data.get('ttt_results', {}).get('optimal_threshold', 'N/A')
    
    print(f"\n📊 ZDR Results:")
    print(f"   TTT ZDR: {ttt_zdr}")
    print(f"   Base ZDR: {base_zdr}")
    print(f"   TTT Threshold: {ttt_threshold}")
    
    if ttt_zdr == 0.0:
        print(f"\n⚠️  ZDR is ZERO - Check logs for diagnostic output!")
        print(f"   Look for: 'DIAGNOSTIC: Zero-day attack probabilities'")
        print(f"   And: 'ROOT CAUSE: Zero-day probabilities are X LOWER'")
else:
    print("❌ No performance metrics file found!")



