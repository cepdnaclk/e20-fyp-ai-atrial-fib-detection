import wfdb
import numpy as np

rec = wfdb.rdrecord("/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/data/physionet/ptbdb/patient001/r01")

print("Signal array:", rec.p_signal[:5])
print("Units:", rec.units)
print("Gain:", rec.adc_gain)
