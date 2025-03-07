# ☁️ azure-ml-pipeline

Serverless ML inference pipeline built on **Azure Functions** and **Azure Blob Storage**. Upload a CSV → trigger preprocessing → run model → write predictions back to Blob.

## Architecture

```
Blob Storage (input/)
        │
        ▼
  Azure Function (BlobTrigger)
        │
        ├── preprocess()
        ├── predict()
        └── store results → Blob Storage (output/)
```

## Features

- **BlobTrigger** — automatically fires when new data lands in `input/` container
- **Sklearn model** loaded once at cold-start via lazy singleton
- **Structured logging** with Azure Application Insights integration
- **Schema validation** with Pydantic before inference
- **Unit tests** with `azure-functions` mock fixtures

## Project structure

```
functions/
├── blob_trigger/
│   ├── __init__.py          # Azure Function entry point
│   └── function.json        # Binding configuration
├── shared/
│   ├── preprocessing.py     # Feature pipeline
│   ├── model_loader.py      # Lazy model singleton
│   └── schemas.py           # Pydantic validation
├── tests/
│   ├── test_preprocessing.py
│   └── test_trigger.py
host.json
local.settings.json.example
requirements.txt
```

## Quick start (local)

```bash
pip install -r requirements.txt
func start   # Azure Functions Core Tools
```

## Deployment

```bash
func azure functionapp publish <YOUR_APP_NAME>
```

## Tech stack

Azure Functions · Azure Blob Storage · Scikit-learn · Pandas · Pydantic · Application Insights

## License

MIT


## Installation

```bash
git clone https://github.com/danielmaddaleno/azure-ml-pipeline.git
cd azure-ml-pipeline
pip install -r requirements.txt
```

## Usage

See `docs/` for detailed examples.
