# TFLearn introduction (legacy)

> **Unmaintained.** These notebooks use [TFLearn](http://tflearn.org/) with
> TensorFlow 1.0 and Python 3.5. TFLearn and TensorFlow 1.x do not run on current
> Python versions, so this folder is kept for reference and is **not** covered by CI.
> The maintained introduction is the PyTorch version in
> [`intro-to-pytorch/`](../../intro-to-pytorch).

## Contents

- `TFLearn_Digit_Recognition*.ipynb`: MNIST digit classifier with TFLearn.
- `TFLearn_Sentiment_Analysis*.ipynb`: IMDB-style sentiment analysis with TFLearn
  (uses `reviews.txt` and `labels.txt`).
- Each notebook has a `_Solution` counterpart.

## Running them

The pins in `requirements.txt` are the original 2017 versions (Python 3.5,
TensorFlow 1.0.0, TFLearn 0.3). The easiest way to get that environment today is Docker:

```bash
docker build -t dl-tflearn-legacy .
```

```bash
docker run --rm -p 8888:8888 dl-tflearn-legacy
```

Then open the URL printed in the container log.

Notes:

- The Dockerfile has not been built or tested in the environment where the PyTorch port
  was done (no Docker daemon was available). If the build fails, an old Python
  (3.5/3.6) virtualenv with `pip install -r requirements.txt` is the fallback.
- TFLearn downloads MNIST from the original host on first use; that download may no
  longer work.
- The state of the repository before the port is also tagged `legacy-tflearn`
  (created locally at commit `1d90117`).
