import sys
import traceback
from pathlib import Path

out_path = Path('decision_import_check.txt')
try:
    # ensure output file is clean
    if out_path.exists():
        out_path.unlink()
    out = out_path.open('w', encoding='utf-8')

    base = Path('decision-service') / 'src'
    target = base / 'models' / 'decision_models.py'
    print(f'Checking file: {target}', file=out)

    if not target.exists():
        print('MISSING_FILE', file=out)
        raise SystemExit(2)

    import importlib.util
    spec = importlib.util.spec_from_file_location('decision_models_check', str(target))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        print('IMPORT_OK', file=out)
        # print a few attributes to confirm content
        names = [n for n in dir(module) if not n.startswith('_')][:20]
        print('TOP_SYMBOLS:' + ','.join(names), file=out)
    except Exception:
        traceback.print_exc(file=out)
        raise
finally:
    try:
        out.close()
    except Exception:
        pass

print('Wrote results to decision_import_check.txt')
