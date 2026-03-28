import os
from pathlib import Path

parent_dir = Path(__file__).resolve().parent

def count_files(directory: str) -> int:
    """Count the number of files (not subdirectories) in a given directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    return sum(1 for entry in os.scandir(directory) if entry.is_file())


if __name__ == "__main__":
      
    sub_dir_1 = parent_dir / "processed_data" / "ground_truth" # Change to your first subdirectory name
    sub_dir_2 = parent_dir / "processed_data" / "input_bev"  # Change to your second subdirectory name

    count1 = count_files(sub_dir_1)
    count2 = count_files(sub_dir_2)

    print(f"Files in '{sub_dir_1}': {count1}")
    print(f"Files in '{sub_dir_2}': {count2}")

    if count1 == count2:
        print("Both directories have the same number of files.")
    else:
        print(f"Mismatch! Difference: {abs(count1 - count2)} file(s).")
    
if __name__ != "__main__":
    print(f"{parent_dir}")