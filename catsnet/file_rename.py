"""Renames all files in a directory to 'mask_0.png' where 0 is replaced with the number
of the file as it appears in the directory"""

from pathlib import Path
import sys

# Check that the correct number of arguments have been passed
if len(sys.argv) != 3:
    print("Usage: python file_rename.py <path> <file_extension>")
    sys.exit()

# Fetch the first argument as the path to the directory
path = Path(sys.argv[1])
if not path.exists():
    print("Path does not exist: ", path)
    sys.exit()

# Get the file extension from the second argument
file_ext = sys.argv[2]
# Ensure that the file extension starts with a period
if file_ext[0] != ".":
    file_ext = "." + file_ext

i = 0

files = list(path.glob("*" + file_ext))
files.sort()

if len(files) == 0:
    print("No files found")
    sys.exit()
else:
    for file in files:
        print(file.name)
        new_filename = path / f"mask_{i}{file_ext}"
        file.rename(new_filename)
        i += 1
