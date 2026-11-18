import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

# Set DL_SMOKE=1 to train for 1 epoch instead of 2 (used for quick automated checks).
SMOKE = os.environ.get('DL_SMOKE') == '1'

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


class Split:
    """Images (flattened, 0-1) and one-hot labels for one part of MNIST."""

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
        labels = F.one_hot(ds.targets[start:stop], 10).float()
        return Split(images.to(device), labels.to(device))

    return MNIST(prepare(train, 0, 55000), prepare(train, 55000, None), prepare(test))


def hist_dist(title, distribution_tensor, hist_range=(-4, 4)):
    """
    Display histogram of a tensor of values drawn from a distribution
    """
    values = distribution_tensor.detach().cpu().numpy()

    plt.title(title)
    plt.hist(values, np.linspace(*hist_range, num=len(values) // 2))
    plt.show()


def _get_loss_acc(dataset, weights):
    """
    Get losses and validation accuracy of example neural network
    """
    batch_size = 128
    epochs = 1 if SMOKE else 2
    learning_rate = 0.001

    # Copy the starting weights so every call starts from exactly the weights we were given
    weights = [w.clone().to(device).requires_grad_() for w in weights]
    biases = [
        torch.zeros(256, device=device, requires_grad=True),
        torch.zeros(128, device=device, requires_grad=True),
        torch.zeros(dataset.train.labels.shape[1], device=device, requires_grad=True),
    ]

    def network(features):
        layer_1 = torch.relu(features @ weights[0] + biases[0])
        layer_2 = torch.relu(layer_1 @ weights[1] + biases[1])
        return layer_2 @ weights[2] + biases[2]

    optimizer = torch.optim.Adam(weights + biases, lr=learning_rate)

    # Measurements use for graphing loss
    loss_batch = []

    batch_count = int(dataset.train.num_examples / batch_size)

    # The training cycle
    for epoch_i in range(epochs):
        for batch_i in range(batch_count):
            batch_features, batch_labels = dataset.train.next_batch(batch_size)

            optimizer.zero_grad()
            loss = F.cross_entropy(network(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())

    with torch.no_grad():
        predictions = network(dataset.validation.images).argmax(dim=1)
        valid_acc = (predictions == dataset.validation.labels.argmax(dim=1)).float().mean().item()

    return loss_batch, valid_acc


def compare_init_weights(
        dataset,
        title,
        weight_init_list,
        plot_n_batches=100):
    """
    Plot loss and print stats of weights using an example neural network
    """
    colors = ['r', 'b', 'g', 'c', 'y', 'k']
    label_accs = []
    label_loss = []

    assert len(weight_init_list) <= len(colors), 'Too many inital weights to plot'

    for i, (weights, label) in enumerate(weight_init_list):
        loss, val_acc = _get_loss_acc(dataset, weights)

        plt.plot(loss[:plot_n_batches], colors[i], label=label)
        label_accs.append((label, val_acc))
        label_loss.append((label, loss[-1]))

    plt.title(title)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    epochs = 1 if SMOKE else 2
    print('After {} Batches ({} Epochs):'.format(epochs * int(dataset.train.num_examples / 128), epochs))
    print('Validation Accuracy')
    for label, val_acc in label_accs:
        print('  {:7.3f}% -- {}'.format(val_acc*100, label))
    print('Loss')
    for label, loss in label_loss:
        print('  {:7.3f}  -- {}'.format(loss, label))
