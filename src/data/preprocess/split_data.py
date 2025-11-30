import os
import shutil
import random
from pathlib import Path
import glob
import numpy as np

def count_files_in_folders(base_dir="/home/tqlong/qtung/gen-model-boilerplate/data/lits"):
    """
    Count and display the number of .npy files in each folder (healthy, unhealthy, masks)
    and their sample directories.
    
    Args:
        base_dir (str): Base directory containing the folders
    """
    base_path = Path(base_dir)
    folders = ['healthy', 'unhealthy', 'masks']
    
    print("\n📊 File Count Summary:")
    print("=" * 50)
    
    total_files = 0
    
    for folder in folders:
        folder_path = base_path / folder
        if not folder_path.exists():
            print(f"❌ {folder}: Folder not found!")
            continue
        
        # Count files in sample directories
        sample_dirs = [d for d in folder_path.iterdir() if d.is_dir() and d.name.startswith('sample_')]
        sample_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
        
        folder_total = 0
        print(f"\n📁 {folder.upper()}:")
        
        for sample_dir in sample_dirs:
            npy_files = list(sample_dir.glob('*.npy'))
            file_count = len(npy_files)
            folder_total += file_count
            print(f"  {sample_dir.name}: {file_count} files")
        
        # Count files in split directories if they exist
        split_dirs = ['train', 'val', 'test']
        split_total = 0
        for split_name in split_dirs:
            split_path = folder_path / split_name
            if split_path.exists():
                split_files = sum(len(list(d.glob('*.npy'))) for d in split_path.rglob('sample_*') if d.is_dir())
                if split_files > 0:
                    split_total += split_files
                    print(f"  {split_name}: {split_files} files")
        
        if split_total > 0:
            print(f"  Split total: {split_total} files")
        
        total_in_folder = folder_total + split_total
        print(f"  📊 Total in {folder}: {total_in_folder} files")
        total_files += total_in_folder
    
    print(f"\n🎯 GRAND TOTAL: {total_files} .npy files")
    print("=" * 50)

def split_healthy_data():
    """
    Split healthy data into train, validation, and test sets independently.
    
    Structure:
    - healthy: healthy_vol_X_slice_YYY.npy
    
    Ratios: 75% train, 15% val, 10% test
    Seed: 12345
    """
    
    # Fixed parameters
    base_dir = "/home/tqlong/qtung/gen-model-boilerplate/data/lits"
    train_ratio = 0.75
    val_ratio = 0.15 
    test_ratio = 0.1
    seed = 12345
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Splitting HEALTHY data with seed {seed}")
    print(f"Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    base_path = Path(base_dir)
    healthy_path = base_path / 'healthy'
    
    # Verify healthy folder exists
    if not healthy_path.exists():
        raise ValueError(f"Healthy folder {healthy_path} does not exist!")
    
    # Collect all healthy files from sample directories
    sample_dirs = [d for d in healthy_path.iterdir() if d.is_dir() and d.name.startswith('sample_')]
    sample_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    print(f"Found {len(sample_dirs)} healthy sample directories")
    
    # Build list of (sample_name, filename) pairs for healthy files
    healthy_files = []
    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        npy_files = list(sample_dir.glob('*.npy'))
        
        for npy_file in npy_files:
            if 'healthy_vol' in npy_file.name and 'slice' in npy_file.name:
                healthy_files.append((sample_name, npy_file.name))
    
    print(f"Found {len(healthy_files)} healthy files to split")
    
    # Shuffle the files to ensure random distribution
    random.shuffle(healthy_files)
    
    # Calculate split indices
    total_files = len(healthy_files)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split the files
    train_files = healthy_files[:train_end]
    val_files = healthy_files[train_end:val_end]
    test_files = healthy_files[val_end:]
    
    print(f"Healthy Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Create splits dictionary
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Process each split for healthy data
    moved_count = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, file_list in splits.items():
        print(f"\nProcessing HEALTHY {split_name} set ({len(file_list)} files)...")
        
        # Create split directory for healthy
        split_folder_path = healthy_path / split_name
        split_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Move files for each healthy file
        for sample_name, filename in file_list:
            # Create sample directory in split
            split_sample_path = split_folder_path / sample_name
            split_sample_path.mkdir(parents=True, exist_ok=True)
            
            # Move healthy file
            src_healthy = healthy_path / sample_name / filename
            dst_healthy = split_sample_path / filename
            
            if src_healthy.exists():
                shutil.move(src_healthy, dst_healthy)
                moved_count[split_name] += 1
                print(f"  ✓ Moved {filename} from healthy/{sample_name}")
            else:
                print(f"  ❌ Warning: {src_healthy} not found!")
    
    # Print move summary for healthy
    print(f"\n📊 Healthy files moved per split:")
    for split_name in ['train', 'val', 'test']:
        print(f"  {split_name}: {moved_count[split_name]} files")
    
    # Print detailed summary for healthy
    print(f"\n=== Healthy Split Summary ===")
    for split_name in ['train', 'val', 'test']:
        split_path = healthy_path / split_name
        if split_path.exists():
            total_files = sum(len(list(d.glob('*.npy'))) for d in split_path.rglob('sample_*') if d.is_dir())
            sample_count = len([d for d in split_path.iterdir() if d.is_dir() and d.name.startswith('sample_')])
            print(f"  {split_name.upper()}: {sample_count} samples, {total_files} .npy files")
    
    # Clean up empty sample directories in healthy
    print("\nCleaning up empty healthy directories...")
    for sample_dir in healthy_path.iterdir():
        if sample_dir.is_dir() and sample_dir.name.startswith('sample_'):
            if not any(sample_dir.iterdir()):  # Empty directory
                sample_dir.rmdir()
                print(f"  Removed empty directory: healthy/{sample_dir.name}")
    
    print("\n✅ Healthy data splitting completed!")
    print("📁 Healthy files moved to: healthy/[train|val|test]/sample_X/")
    print("💾 Original healthy files moved to save disk space")

def split_lits_data():
    """
    Split LiTS data into train, validation, and test sets.
    Ensures mask and unhealthy files are paired correctly by matching slice numbers.
    
    Structure:
    - healthy: healthy_vol_X_slice_YYY.npy
    - unhealthy: unhealthy_vol_X_slice_YYY.npy  
    - masks: unhealthy_mask_X_slice_YYY.npy
    
    Ratios: 75% train, 15% val, 10% test
    Seed: 12345
    """
    
    # Fixed parameters
    base_dir = "/home/tqlong/qtung/gen-model-boilerplate/data/lits"
    train_ratio = 0.75
    val_ratio = 0.15 
    test_ratio = 0.1
    seed = 12345
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Splitting LiTS data with seed {seed}")
    print(f"Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    # Show initial file count
    count_files_in_folders(base_dir)
    
    base_path = Path(base_dir)
    folders = ['healthy', 'unhealthy', 'masks']
    
    # Verify folders exist
    for folder in folders:
        folder_path = base_path / folder
        if not folder_path.exists():
            raise ValueError(f"Folder {folder_path} does not exist!")
    
    # Get all unhealthy files to ensure mask-unhealthy pairing
    unhealthy_path = base_path / 'unhealthy'
    all_unhealthy_files = []
    
    # Collect all unhealthy files with their corresponding sample directories
    sample_dirs = [d for d in unhealthy_path.iterdir() if d.is_dir() and d.name.startswith('sample_')]
    sample_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    # Build list of (sample_name, file_stem) pairs for unhealthy files
    unhealthy_pairs = []
    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        npy_files = list(sample_dir.glob('*.npy'))
        
        for npy_file in npy_files:
            # Extract slice info from filename: unhealthy_vol_X_slice_YYY.npy
            file_stem = npy_file.stem  # e.g., unhealthy_vol_0_slice_046
            if 'unhealthy_vol' in file_stem and 'slice' in file_stem:
                unhealthy_pairs.append((sample_name, file_stem))
    
    print(f"Found {len(unhealthy_pairs)} unhealthy files to split")
    
    # Shuffle the pairs to ensure random distribution
    random.shuffle(unhealthy_pairs)
    
    # Calculate split indices
    total_files = len(unhealthy_pairs)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Split the pairs
    train_pairs = unhealthy_pairs[:train_end]
    val_pairs = unhealthy_pairs[train_end:val_end]
    test_pairs = unhealthy_pairs[val_end:]
    
    print(f"Split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Create splits dictionary
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    # Process each split
    moved_count = {'train': 0, 'val': 0, 'test': 0}
    
    for split_name, file_pairs in splits.items():
        print(f"\nProcessing {split_name} set ({len(file_pairs)} files)...")
        
        # Create split directories
        for folder in folders:
            split_folder_path = base_path / folder / split_name
            split_folder_path.mkdir(parents=True, exist_ok=True)
        
        # Move files for each pair
        for sample_name, file_stem in file_pairs:
            # Create sample directories in splits
            for folder in folders:
                split_sample_path = base_path / folder / split_name / sample_name
                split_sample_path.mkdir(parents=True, exist_ok=True)
            
            files_moved_in_pair = 0
            
            # 1. Move unhealthy file first
            src_unhealthy = base_path / 'unhealthy' / sample_name / f"{file_stem}.npy"
            dst_unhealthy = base_path / 'unhealthy' / split_name / sample_name / f"{file_stem}.npy"
            
            if src_unhealthy.exists():
                shutil.move(src_unhealthy, dst_unhealthy)
                files_moved_in_pair += 1
                print(f"  ✓ Moved {file_stem}.npy from unhealthy/{sample_name}")
            else:
                print(f"  ❌ Warning: {src_unhealthy} not found!")
                continue  # Skip this pair if unhealthy file not found
            
            # 2. Move corresponding mask file
            # Convert unhealthy_vol_X_slice_YYY to unhealthy_mask_X_slice_YYY
            mask_stem = file_stem.replace('unhealthy_vol', 'unhealthy_mask')
            src_mask = base_path / 'masks' / sample_name / f"{mask_stem}.npy"
            dst_mask = base_path / 'masks' / split_name / sample_name / f"{mask_stem}.npy"
            
            if src_mask.exists():
                shutil.move(src_mask, dst_mask)
                files_moved_in_pair += 1
                print(f"  ✓ Moved {mask_stem}.npy from masks/{sample_name}")
            else:
                print(f"  ❌ Warning: Corresponding mask {src_mask} not found!")
            
            # 3. Move ALL corresponding healthy files (not just one)
            # Extract volume and slice info
            parts = file_stem.split('_')
            if len(parts) >= 4:  # unhealthy_vol_X_slice_YYY
                vol_num = parts[2]
                slice_num = parts[4]
                
                # Find all healthy files with same volume and slice
                healthy_pattern = f"healthy_vol_{vol_num}_slice_{slice_num}"
                src_healthy_dir = base_path / 'healthy' / sample_name
                
                if src_healthy_dir.exists():
                    # Find all healthy files matching the pattern
                    healthy_files = list(src_healthy_dir.glob(f"{healthy_pattern}*.npy"))
                    
                    for healthy_file in healthy_files:
                        dst_healthy = base_path / 'healthy' / split_name / sample_name / healthy_file.name
                        shutil.move(healthy_file, dst_healthy)
                        files_moved_in_pair += 1
                        print(f"  ✓ Moved {healthy_file.name} from healthy/{sample_name}")
            
            moved_count[split_name] += files_moved_in_pair
    
    # Print move summary
    print(f"\n📊 Files moved per split:")
    for split_name in ['train', 'val', 'test']:
        print(f"  {split_name}: {moved_count[split_name]} files")
    
    # Print summary
    print(f"\n=== Split Summary ===")
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()} SET:")
        for folder in folders:
            split_path = base_path / folder / split_name
            if split_path.exists():
                total_files = sum(len(list(d.glob('*.npy'))) for d in split_path.rglob('sample_*') if d.is_dir())
                sample_count = len([d for d in split_path.iterdir() if d.is_dir() and d.name.startswith('sample_')])
                print(f"  {folder}: {sample_count} samples, {total_files} .npy files")
    
    # Clean up empty sample directories
    print("\nCleaning up empty directories...")
    for folder in folders:
        folder_path = base_path / folder
        for sample_dir in folder_path.iterdir():
            if sample_dir.is_dir() and sample_dir.name.startswith('sample_'):
                if not any(sample_dir.iterdir()):  # Empty directory
                    sample_dir.rmdir()
                    print(f"  Removed empty directory: {folder}/{sample_dir.name}")
    
    print("\n✅ Data splitting completed!")
    print("🔗 Mask and unhealthy files are properly paired by slice numbers") 
    print("📁 Files moved to: healthy/[split]/sample_X/, unhealthy/[split]/sample_X/, masks/[split]/sample_X/")
    print("💾 Original files deleted to save disk space")
    
    # Show final file count
    count_files_in_folders(base_dir)

if __name__ == "__main__":
    # You can run just the count function
    count_files_in_folders()
    
    # Run healthy data splitting only
    # split_healthy_data()
    
    # Run the LiTS data splitting (unhealthy + masks)
    # split_lits_data()
