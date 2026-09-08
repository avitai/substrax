# Tracking

`substrax.tracking` records scalars, images, histograms, text and hyperparameters per
step, to the console, to files, to Weights & Biases or to MLflow, behind one `Logger`
interface.

```python
from substrax.tracking import FileLogger, WandbLogger

logger = FileLogger("run-1", log_dir="logs/run-1")
logger.log_scalars({"loss": 0.42, "accuracy": 0.9}, step=100)
logger.log_image("samples", images, step=100)
logger.log_hyperparams({"learning_rate": 1e-3})
logger.close()

wandb_logger = WandbLogger("run-1", project="my-project")
```

| Backend | Extra | Where things go |
| --- | --- | --- |
| `ConsoleLogger` | none | `logging` lines on stdout |
| `FileLogger` | `plots` for figures | Lines to a `.log` file, scalars to a long-form CSV (`timestamp,step,name,value`), images and histograms to PNGs, texts and hyperparameters to text files |
| `WandbLogger` | `wandb` | A W&B run; `log_code` and `log_model` upload code and model artifacts |
| `MLFlowLogger` | `mlflow`, `plots` for figures | An MLflow run; images, histograms and texts as run artifacts |

Importing `substrax.tracking` imports neither SDK. A backend loads its SDK when
constructed and raises `ImportError` naming the extra to install:

```bash
uv add 'substrax[wandb]'
```

::: substrax.tracking
