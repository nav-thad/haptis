from pathlib import Path
import random
random.seed(42)

for cls in ['noise', 'voice', 'alarm']:
    files = list(Path('data').glob(f'{cls}/*.wav'))
    random.shuffle(files)
    
    split = int(0.8 * len(files))
    train_files = files[:split]
    test_files = files[split:]
    
    # Move test files
    Path('data_test').mkdir(exist_ok=True)
    (Path('data_test') / cls).mkdir(exist_ok=True)
    
    for f in test_files:
        f.rename(Path('data_test') / cls / f.name)
    
    print(f'{cls}: {len(files)} → train:{len(train_files)} test:{len(test_files)}')