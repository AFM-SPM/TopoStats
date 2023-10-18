"""Renames all files in a directory to 'mask_0.png' where 0 is replaced with the number
of the file as it appears in the directory"""

from pathlib import Path

path = Path("/Users/sylvi/topo_data/hariborings/training_data/on_target_labelled/")
i = 0

files = list(path.glob("*.png"))
files.sort()
for file in files:
    print(file.name)
    new_filename = path / f"mask_{i}.png"
    file.rename(new_filename)
    i += 1
