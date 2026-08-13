# FinTech Project - Part B

PortFoYou is an investor-facing Streamlit application for comparing systematically
managed funds, reviewing fund fact sheets, testing allocation scenarios, and
exploring equity-sector news sentiment.

- Live application: https://z5707852projectb-crdypbfgqdwsyx35aw7t99.streamlit.app
- Public repository: https://github.com/JW5707852/z5707852_projectB
- App entrypoint: `streamlit_app.py`

## How to run

    pip install -r requirements.txt -r requirements-dev.txt   # dev adds nltk (VADER)
    python scripts/run_part_b.py            # reproduces your results into results/
    streamlit run streamlit_app.py          # runs the app locally

Load raw data through src/data_access.py (see context/DATA_GUIDE.md); never commit
raw data. The deployed app, by contrast, reads your precomputed artifacts from
results/ - those ARE committed.

## What is here

- streamlit_app.py    the app entrypoint (repo root)
- .streamlit/         app config
- PROJECT_BRIEF.md    the full assignment brief for your course (read this first)
- src/                your code (data_access is provided; portfolios/sentiment/fusion are yours)
- scripts/            runnable scripts that reproduce your results
- results/            your outputs: figures in results/figures/, tables in results/tables/, app data artifacts in results/data/
- context/            provided data guide and project context (do not edit)
- report/             your report - see report/OUTLINE.md (author in Word, submit report.pdf)
- ai/                 your prompt logs and AI notes
- requirements-dev.txt build/repro-only deps (nltk); keep them out of the deployed app
- AGENTS.md / CLAUDE.md   replace the stub for your tool (you need just one) with your own

## Deploy + hand in

This folder is its own GitHub repository, independent of fins-agent. Before a
release, run:

    python scripts/check_handin.py        # your agent can run this
    # commit your precomputed app artifacts under results/ (the app reads them)
    # commit the verified changes and push the main branch

The existing Streamlit Community Cloud application follows the repository's
`main` branch and uses `streamlit_app.py` as its entrypoint.
