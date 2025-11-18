#!/usr/bin/env python3
"""
Comprehensive ECG Clinical Interface Launcher
"""

import webbrowser
from pathlib import Path

html_file = Path(r"D:\research\codes\results\phase3\cofe\comprehensive_clinical_interface.html")

print("🌐 Opening Comprehensive Clinical Interface...")
print(f"   📁 File: {html_file}")
print()
webbrowser.open(f'file://{html_file.absolute()}')
print("   ✅ Interface opened in browser!")
print()
print("📖 Interface Features:")
print("   🧬 Tab 1: CoFE Counterfactuals with decision boundary control")
print("   🎛️ Tab 2: Feature Manipulation (Direct + Latent methods)")
print("   📊 Tab 3: 3-Way Method Comparison")
print()
print("   Close browser tab when done.")
