#!/usr/bin/env python3
"""
ECG Counterfactual Clinical Interface Launcher
"""

import webbrowser
from pathlib import Path

# Path to HTML file
html_file = Path(r"D:\research\codes\results\phase3\cofe\clinical_interface.html")

# Open in default browser
print("🌐 Opening Clinical Interface in browser...")
print(f"   File: {html_file}")
webbrowser.open(f'file://{html_file.absolute()}')
print("   ✅ Interface opened!")
print()
print("📖 Instructions:")
print("   • Select ECG pairs from the dropdown")
print("   • Use sliders to zoom and navigate")
print("   • Toggle views with checkboxes")
print("   • Hover over plots for details")
print()
print("   Close the browser tab when done.")
