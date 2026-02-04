# 

Utils and methods for weight-space merging.

---

## 🚀 Installation

You can install this project using [`uv`](https://github.com/astral-sh/uv):

```sh
uv sync
```

---

## 📂 Project Structure

```
/
├── src/                     # Source directory
│   ├── model-merging/  # Main package
│   │   ├── __init__.py
│   │   ├── main.py
├── pyproject.toml           # Package configuration
├── README.md                # This file
└── LICENSE                  # License information
```

---

## Multi-Task Merging

Use `conf/multitask.yaml` to define the models you want to merge and the tasks you will evaluate the merged model on. Then run

```sh
uv run scripts/evaluate_multi_task_merging.py
```

If you want to define a new merging method, create a new class in `src/model_merging/merger/` and a corresponding config in `conf/merger`. Then change the `merger` field in the `multitask.yaml` config.
Results can also be reproduced using the notebook in therepo


---

## 📜 License

This project is licensed under the **MIT** License. See [LICENSE](LICENSE) for more details.
