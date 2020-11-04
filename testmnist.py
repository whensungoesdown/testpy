import torch
import torchvision
import matplotlib.pyplot as plt
import util
import torch.nn.functional as F

batch_size_test = 1000

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./mnistdata', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)




# Specify a path
PATH = "saves/model_after_weight_sharing.ptmodel"

# Load
model = torch.load(PATH)
model.eval()

print(model)
util.print_model_parameters(model)



examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


fig = plt.figure()
#for i in range(6):
#  plt.subplot(2,3,i+1)
for i in range(1):
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

plt.show()


#test
def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print(input)
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print(output)
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())


model.fc1.register_forward_hook(printnorm)
model.fc2.register_forward_hook(printnorm)
model.fc3.register_forward_hook(printnorm)
#


result = model(example_data[0])
pred = result.data.max(1, keepdim=True)[1] # get the index of the max log-probability
print("\npruned model predict:")
print(pred)
print("\n")








# Specify a path
PATH_ORIG = "saves/initial_model.ptmodel"

# Load
model_orig = torch.load(PATH_ORIG)
model_orig.eval()

print(model_orig)
util.print_model_parameters(model_orig)

model_orig.fc1.register_forward_hook(printnorm)
model_orig.fc2.register_forward_hook(printnorm)
model_orig.fc3.register_forward_hook(printnorm)

result_orig = model_orig(example_data[0])
pred_orig = result_orig.data.max(1, keepdim=True)[1] # get the index of the max log-probability
print("\noriginal model predict:")
print(pred_orig)
print("\n")







def test():
    device = torch.device('cpu')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


#accuracy = test()
#print(accuracy)

