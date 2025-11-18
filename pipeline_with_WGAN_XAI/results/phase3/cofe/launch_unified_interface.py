#!/usr/bin/env python3
"""
Unified ECG Feature Manipulation Interface Launcher
"""

import webbrowser
from pathlib import Path

html_file = Path(r"D:\research\codes\results\phase3\cofe\unified_interface.html")

print("🌐 Opening Unified Interface in browser...")
print(f"   File: {html_file}")
webbrowser.open(f'file://{html_file.absolute()}')
print("   ✅ Interface opened!")
print()
print("📖 Features:")
print("   • Compare Direct vs Latent manipulation side-by-side")
print("   • Real-time classification prediction")
print("   • Comprehensive comparison table")
print("   • Overlay visualization")
print()
