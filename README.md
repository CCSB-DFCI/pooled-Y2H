# nanopore_y2h

Development of a pooled Y2H variant edgotyping approach using long-read sequencing.

## Installation

We're using uv for dependency / python version etc.

See here: https://docs.astral.sh/uv
On mac/linux you can install uv with:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
You probably need to restart or source your shell after installing.

Then, to install the dependencies for this project, run:
```
uv sync
```
This will create a virtual environment (`.venv/`) and install the exact dependency versions.

You can then run scripts using the exact python and library versions like this:
```
uv run python src/some_script.py
```

Or activate the virtual environment with:
```
source .venv/bin/activate
```

## Input data

- The fastq files from the nanopore sequencing should be availble from the IGVF data portal. 