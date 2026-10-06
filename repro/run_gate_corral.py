"""Run GATE's own corralFreeNoiseSimulator.py unmodified. Their script calls
pegasusQSVCTrain(..., feature_map, "0") but GSI.pegasusQSVCTrain takes 5 args; this wrapper
only drops the stray 6th arg (a GPU id used by the *NoiseSimulator variant)."""
import runpy, sys
sys.path.insert(0, ".")
import GSI
_orig = GSI.pegasusQSVCTrain
GSI.pegasusQSVCTrain = lambda *a, **k: _orig(*a[:5], **k)
runpy.run_path("FreeNoiseSimulatorCode/corralFreeNoiseSimulator.py", run_name="__main__")
