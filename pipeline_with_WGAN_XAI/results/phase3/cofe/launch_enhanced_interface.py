#!/usr/bin/env python3
"""
Enhanced Interface with Predictions Launcher
"""

import webbrowser
from pathlib import Path

html_file = Path(r"D:\research\codes\results\phase3\cofe\enhanced_interface_with_predictions.html")

print("🌐 Opening Enhanced Interface with Predictions...")
print(f"   📁 File: {html_file}")
print()
webbrowser.open(f'file://{html_file.absolute()}')
print("   ✅ Interface opened!")
print()
print("✨ Features:")
print("   ✅ Real-time predictions for ALL signals")
print("   ✅ Decision boundary actually affects generation")
print("   ✅ Feature manipulation shows classification changes")
print("   ✅ Side-by-side prediction comparison")
print()
