# Legacy environment for the TFLearn notebooks (unmaintained stack, 2017).
# Python 3.5 + TensorFlow 1.0 + TFLearn 0.3, as pinned in requirements.txt.
#
# Build (from this folder):   docker build -t dl-tflearn-legacy .
# Run:                        docker run --rm -p 8888:8888 dl-tflearn-legacy
# Then open the URL with the token printed in the container log.
FROM python:3.5-slim

WORKDIR /notebooks

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir=/notebooks"]
