import torch
import torchvision
import matplotlib.pyplot as plt


batch_size_test = 1000

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./mnistdata', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

plt.show()


# Specify a path
PATH = "saves/model_after_weight_sharing.ptmodel"

# Load
model = torch.load(PATH)
model.eval()
