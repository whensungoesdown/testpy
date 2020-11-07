import torch
import torchvision
import matplotlib.pyplot as plt
import util
import torch.nn.functional as F
import torch.tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


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


#fig = plt.figure()
##for i in range(6):
##  plt.subplot(2,3,i+1)
#for i in range(1):
#  plt.tight_layout()
#  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#  plt.title("Ground Truth: {}".format(example_targets[i]))
#  plt.xticks([])
#  plt.yticks([])
#fig
#
#plt.show()




#global fc3input

#test
def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print(input)
    print('output: ', type(output))
    print('output[0]: ', type(output[0]))
    print('output.data:', output.data)
#    print('output[0].data:', output[0].data)
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printnorm_fc2_prune(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print(input)
    print('output: ', type(output))
    print('output[0]: ', type(output[0]))
    print('output.data:', output.data)
#    print('output[0].data:', output[0].data)
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('!!!! copy tensor')
    global fc2_prune_input
    fc2_prune_input = input[0].clone().detach()

def printnorm_fc2_orig(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print(input)
    print('output: ', type(output))
    print('output[0]: ', type(output[0]))
    print('output.data:', output.data)
#    print('output[0].data:', output[0].data)
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('!!!! copy tensor')
    global fc2_orig_input
    fc2_orig_input = input[0].clone().detach()



def printnorm_fc3_prune(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print(input)
    print('output: ', type(output))
    print('output[0]: ', type(output[0]))
    print('output.data:', output.data)
#    print('output[0].data:', output[0].data)
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('!!!! copy tensor')
    global fc3_prune_input
    fc3_prune_input = input[0].clone().detach()

def printnorm_fc3_orig(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print(input)
    print('output: ', type(output))
    print('output[0]: ', type(output[0]))
    print('output.data:', output.data)
#    print('output[0].data:', output[0].data)
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('!!!! copy tensor')
    global fc3_orig_input
    fc3_orig_input = input[0].clone().detach()
    

model.fc1.register_forward_hook(printnorm)
model.fc2.register_forward_hook(printnorm_fc2_prune)
model.fc3.register_forward_hook(printnorm_fc3_prune)
#


#result = model(example_data[0])
#pred = result.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#print("\npruned model predict:")
#print(pred)
#print("\n")








# Specify a path
PATH_ORIG = "saves/initial_model.ptmodel"

# Load
model_orig = torch.load(PATH_ORIG)
model_orig.eval()

print(model_orig)
util.print_model_parameters(model_orig)

model_orig.fc1.register_forward_hook(printnorm)
model_orig.fc2.register_forward_hook(printnorm_fc2_orig)
model_orig.fc3.register_forward_hook(printnorm_fc3_orig)

#result_orig = model_orig(example_data[0])
#pred_orig = result_orig.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#print("\noriginal model predict:")
#print(pred_orig)
#print("\n")



for i in range(1000):

    result = model(example_data[i])
    pred = result.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    
    result_orig = model_orig(example_data[i])
    pred_orig = result_orig.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    
    
    if pred == pred_orig:
        #print('fc3input type:', type(fc3input))
        #print('fc2_prune_input :', fc2_prune_input)
        #print('fc2_orig_input :', fc2_orig_input)
        
        fc2_input_diffabs = torch.abs(fc2_orig_input - fc2_prune_input);
        
        #print('fc2_input_diffabs :', fc2_input_diffabs)
        
        
        #fc3_input_diffabs_histc = torch.histc(fc3_input_diffabs, 100, 0, 100)
        #print('fc3_input_diffabs_histc :', fc3_input_diffabs_histc)
        
        writer.add_histogram('fc2/prune_input', fc2_prune_input, global_step=example_targets[i], bins='tensorflow')
        writer.add_histogram('fc2/orig_input', fc2_orig_input, global_step=example_targets[i], bins='tensorflow')
        writer.add_histogram('fc2/input_diffabs', fc2_input_diffabs, global_step=example_targets[i], bins='tensorflow')
        
        
        #print('fc3_prune_input :', fc3_prune_input)
        #print('fc3_orig_input :', fc3_orig_input)
        
        fc3_input_diffabs = torch.abs(fc3_orig_input - fc3_prune_input);
        
        #print('fc3_input_diffabs :', fc3_input_diffabs)
        
        
        #fc3_input_diffabs_histc = torch.histc(fc3_input_diffabs, 100, 0, 100)
        #print('fc3_input_diffabs_histc :', fc3_input_diffabs_histc)
        
        writer.add_histogram('fc3/prune_input', fc3_prune_input, global_step=example_targets[i], bins='tensorflow')
        writer.add_histogram('fc3/orig_input', fc3_orig_input, global_step=example_targets[i], bins='tensorflow')
        writer.add_histogram('fc3/input_diffabs', fc3_input_diffabs, global_step=example_targets[i], bins='tensorflow')


writer.close()



#def count_elements(seq) -> dict:
#    """Tally elements from `seq`."""
#    hist = {}
#    for i in seq:
#        hist[i] = hist.get(i, 0) + 1
#    return hist
#
#
#def ascii_histogram(seq) -> None:
#    """A horizontal frequency-table/histogram plot."""
#    counted = count_elements(seq)
#    for k in sorted(counted):
#        print('{0:5d} {1}'.format(k, '+' * counted[k]))
#        
#
#ascii_histogram(fc3_input_diffabs_histc.data)



def ascii_histogram(seq) -> None:
    """A horizontal frequency-table/histogram plot."""
    for k in seq:
        print('{0:5d} {1}'.format(k, '+' * k))

#ascii_histogram(fc3_input_diffabs_histc.long())









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

