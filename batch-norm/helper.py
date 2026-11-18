import torch
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


class Split:
    """Images and integer class labels for one part of MNIST."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.num_examples = len(images)

    def next_batch(self, batch_size):
        """Return a random batch of `batch_size` examples."""
        idx = torch.randint(0, self.num_examples, (batch_size,), device=self.images.device)
        return self.images[idx], self.labels[idx]


class MNIST:
    """MNIST split into train (55,000), validation (5,000) and test (10,000)."""

    def __init__(self, train, validation, test):
        self.train, self.validation, self.test = train, validation, test


def load_mnist(root='data', reshape=True):
    """
    Download (if needed) and load MNIST as tensors on the active device.

    :param reshape: if True, images are flattened to 784 values; otherwise they have shape (1, 28, 28)
    """
    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)

    def prepare(ds, start=None, stop=None):
        images = ds.data[start:stop].float().div(255)
        images = images.reshape(len(images), -1) if reshape else images.unsqueeze(1)
        return Split(images.to(device), ds.targets[start:stop].to(device))

    return MNIST(prepare(train, 0, 55000), prepare(train, 55000, None), prepare(test))
