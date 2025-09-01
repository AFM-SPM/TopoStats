import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #load
    #filters
    #grains
    #stats
    """
    )
    return


@app.cell
def _():
    from pathlib import Path
    from topostats.io import LoadScans, find_files, read_yaml
    return Path, find_files


@app.cell
def _(Path, find_files):
    # Set the base directory to be current working directory of the Notebook
    BASE_DIR = Path().cwd()
    print(BASE_DIR)
    # Alternatively if you know where your files area comment the above line and uncomment the below adjust it for your use.
    # BASE_DIR = Path("/path/to/where/my/files/are")
    # Adjust the file extension appropriately.
    FILE_EXT = ".spm"
    # Search for *.spm files one directory level up from the current notebooks
    image_files = find_files(base_dir=BASE_DIR.parent / "tests", file_ext=FILE_EXT)
    return (image_files,)


@app.cell
def _(image_files):
    print(image_files)
    return


if __name__ == "__main__":
    app.run()
