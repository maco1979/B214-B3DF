import sys
import traceback
from pathlib import Path
import importlib.util

base = Path('decision-service') / 'src'
out_path = Path('decision_service_import_all.txt')

modules = []
for p in base.rglob('*.py'):
    # skip __init__ files if desired but we'll include them
    modules.append(p)

results = []
with out_path.open('w', encoding='utf-8') as out:
    print(f'Found {len(modules)} python files under {base}', file=out)
    for p in sorted(modules):
        rel = p.relative_to(base)
        name = 'decision.' + '.'.join(rel.with_suffix('').parts)
        print(f'-- Importing {p} as {name}', file=out)
        try:
            spec = importlib.util.spec_from_file_location(name, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print('   OK', file=out)
        except Exception:
            print('   FAILED', file=out)
            traceback.print_exc(file=out)
            results.append((str(p), 'FAILED'))

    print('\nSummary:', file=out)
    total = len(modules)
    failed = len(results)
    print(f'Total: {total}  Failed: {failed}', file=out)
    if failed:
        print('Failures:', file=out)
        for p, s in results:
            print(p, s, file=out)

print(f'Wrote results to {out_path}')
