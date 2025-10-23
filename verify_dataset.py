from pathlib import Path
import sys

def main(root='yolo_ready'):
    p = Path(root)
    if not p.exists():
        print('Dataset folder not found:', p.resolve())
        return 1
    print('Root:', p.resolve())
    for sub in ['images/train','images/val','labels/train','labels/val']:
        sp = p / Path(sub)
        print(f"{sub}: exists={sp.exists()}, files={len(list(sp.glob('*.*'))) if sp.exists() else 0}")
    yaml = p / 'data.yaml'
    if yaml.exists():
        print('\ndata.yaml content:')
        print(yaml.read_text())
    else:
        print('\ndata.yaml not found — create one using data_yaml.template as reference')
    return 0

if __name__ == '__main__':
    sys.exit(main())
