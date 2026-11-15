# Curriculum

A suggested order for working through the repository. Each stage builds on the previous one, and every track ends with a project. Stages 3, 4 and 5 are independent of each other, so you can take only the tracks you need.

Status refers to the PyTorch port (see `ported_folders.txt`). Folders not yet ported still use TensorFlow 1.x and the legacy requirements in their own folder.

## Stage 1: Foundations (numpy only)

How a network learns: loss, gradients and the update rule.

1. `linear-regression`: fitting a line, the loss function.
2. `gradient-descent`: gradients and the weight-update rule.
3. `student-admissions`: a first classifier.
4. `sentiment-network`: a network built by hand, and cleaning up input features.
5. **Project:** `first-neural-network`: predict bike rentals. Implement the forward and backward passes yourself.

## Stage 2: The framework and making training work

6. `intro-to-tensorflow`: first steps with a framework (tensors, layers, optimizers). Becomes the PyTorch intro in the port.
7. `intro-to-pytorch/Student_Admissions_PyTorch.ipynb`: the admissions problem again, with a framework doing the gradients.
8. `intro-to-pytorch/Sentiment_Analysis_PyTorch.ipynb`: the sentiment problem again, with a bag-of-words network.
9. `intro-to-pytorch/Digit_Recognition_PyTorch.ipynb`: classifying MNIST digits. (The original TFLearn and Keras versions of steps 7 to 9 are kept in `legacy/`.)
10. `weight-initialization`: why training can fail from the start.
11. `batch-norm`: stabilizing and speeding up training.
12. `tensorboard`: watching the network and its parameters during training.

## Stage 3: Images

13. `image-classification`, **project:** a convolutional network for CIFAR-10.
14. `transfer-learning`: reuse a pretrained VGG to classify flowers.
15. `autoencoder`: compression and denoising with learned representations.
16. `semi-supervised`: using unlabeled data.

## Stage 4: Sequences and text

17. `intro-to-rnns`: a character-level RNN.
18. `embeddings`: word2vec.
19. `sentiment-rnn`: embeddings and an RNN applied to the sentiment task from stage 1.
20. `tv-script-generation`, **project:** generate Simpsons scripts with an RNN.
21. `seq2seq`: encoder-decoder networks.
22. `language-translation`, **project:** English to French translation.

## Stage 5: Generative models and reinforcement learning

23. `gan_mnist`: a simple GAN.
24. `dcgan-svhn`: a convolutional GAN on street-view house numbers.
25. `face_generation`, **project:** a DCGAN on CelebA.
26. `reinforcement`: deep Q-learning on a simple OpenAI Gym game.

## How the stages connect

- Every stage repeats the same loop: data, model, loss, optimizer, evaluation. Stage 1 does it by hand and later stages let the framework do the work.
- Stage 4 revisits sentiment analysis (steps 4, 8 and 19) with progressively more powerful models, so you can compare them.
- Seq2seq and machine translation come last in their track because they combine most of the earlier ideas.
