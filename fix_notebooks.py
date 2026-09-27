import json
from pathlib import Path

files = [Path('cnn.ipynb'), Path('mobilenet.ipynb')]
for path in files:
    data = json.loads(path.read_text(encoding='utf-8'))
    changed = False
    for cell in data.get('cells', []):
        src = ''.join(cell.get('source', []))
        if "layers.Dense(train_data.num_classes, activation='softmax')" in src:
            src = src.replace(
                "layers.Dense(train_data.num_classes, activation='softmax')",
                "layers.Dense(len(train_data.class_indices), activation='softmax')",
            )
            cell['source'] = src.splitlines(keepends=True)
            changed = True
        # clear stale execution trace/errors from prior notebook runs
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
print('updated cnn.ipynb and mobilenet.ipynb')
