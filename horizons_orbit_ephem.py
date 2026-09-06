name: Comet orbital ephemeris JSON

on:
  schedule:
    # Run once a day; adjust time to taste (UTC)
    - cron: "15 3 * * *"
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install astroquery astropy requests

      - name: Generate comet orbital ephemeris JSON
        env:
          # Same env var your main script uses for brightness, if you want to override
          BRIGHT_LIMIT: "15.0"
        run: |
          python horizons_orbit_ephem.py

      - name: Commit and push if changed
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: "Update comet orbital ephemeris JSON"
          file_pattern: data/comets_orbit_ephem.json
