"""Portable entry point: python scripts/run_verification.py train|evaluate|infer ..."""
import runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
if len(sys.argv)<2 or sys.argv[1] not in ('train','evaluate','infer'):
    raise SystemExit('Usage: run_verification.py train|evaluate|infer [arguments]')
module={'train':'train','evaluate':'evaluate','infer':'inference'}[sys.argv.pop(1)]
runpy.run_module('verification.'+module,run_name='__main__')
