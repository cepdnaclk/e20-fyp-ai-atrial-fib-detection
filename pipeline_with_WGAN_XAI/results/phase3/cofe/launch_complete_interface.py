#!/usr/bin/env python3
import webbrowser
from pathlib import Path

html_file = Path(r"D:\research\codes\results\phase3\cofe\complete_interface_with_predictions.html")
print("🌐 Opening Complete Interface with Predictions...")
webbrowser.open(f'file://{html_file.absolute()}')
print("   ✅ Interface opened!")
print()
print("✨ Features:")
print("   Tab 1: CoFE with WORKING threshold control")
print("   Tab 2: Direct + Latent manipulation with predictions")
print("   Tab 3: 3-way comparison with statistics")
print()
