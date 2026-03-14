# Vacant_Lot_Detection

## IMPORTANT NOTES 
- Repo development is done with bare worktree layout, all artifacts are written to a dir above, make sure to change shared repo root if changing this. 

## Notebook Output Stripping (nbstripout)

This repo uses [nbstripout](https://github.com/kynan/nbstripout) to manage notebook outputs in git. The filter runs automatically on commit — you never need to run it manually.

By default, **all cell outputs are stripped**. To preserve a specific cell's output (e.g. a figure or table), add the `keep_output` tag to that cell:

- **VS Code**: click the cell's `...` menu → Add Cell Tag → type `keep_output`
- **Jupyter**: View → Cell Toolbar → Tags → type `keep_output` and click Add tag

### First-time setup (required per clone)

The git filter config lives in `.git/config` (or `.bare/config` for bare repos) and is **not tracked by git**. After cloning, run:

```bash
uv sync
uv run nbstripout --install --attributes .gitattributes
```

For a bare repo + worktrees layout, the filter is written to `.bare/config` and applies to all worktrees. Each worktree still needs its own `uv sync` so that `nbstripout` is available in its `.venv`.

### How it works

- `.gitattributes` (tracked) maps `*.ipynb` to the `nbstripout` filter
- The git config (not tracked) defines the filter command: `uv run python -m nbstripout`
- `uv run` resolves to whichever worktree you're committing from, using that worktree's `.venv`

## Setup

1. Set up [Google ADC](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)
2. Enable the Earth Engine Api through the UI or CLI
    - UI
        ```
        https://console.cloud.google.com/apis/api/earthengine.googleapis.com/metrics?project=<PROJECT>
        ```
    - CLI
        ```
        gcloud services enable earthengine.googleapis.com --project=<PROJECT>
        ```
2. configure roles to Service Usage Consumer + Earth Engine Resource Admin in project https://console.cloud.google.com/iam-admin/iam?project={PROJECT_ID}
3. enabke the earth engine api https://console.cloud.google.com/apis/api/earthengine.googleapis.com/metrics?project=vacant-lot-detection 
4. make sure project is added to gee 
5. register the project https://console.cloud.google.com/earth-engine/configuration;success=true?project=vacant-lot-detection
6. uv run earthengine authenticate --force 
refresh credentials 
