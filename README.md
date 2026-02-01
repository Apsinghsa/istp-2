Usage
-----

This workspace prefers running Python scripts via `uv` so they execute inside the project's configured environment.

Examples:

- Run the plotting script (creates and shows the plot):

  uv run data_plot.py

- Run without showing the plot interactively (saves to `plots/kandhi_bajoura.png` by default):

  uv run data_plot.py --no-show

Notes
-----
- Make sure the project's virtual environment is set up and `uv` is available. Use `uv run <script>` instead of `python <script>` to ensure the correct interpreter and environment are used.
