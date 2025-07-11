import torch

class torchNN(torch.nn.Module):
  #This was created using the example given from pytorch's website which is then modified. https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
  """
  The hand recognition software has 21 datapoints to place the hand. Each of these has 3 cordinates being measured.
  So the datasize per input is [x,y,z]*21 or 63 inputs. Therefore the neural network will need a starting neuron size of 93 to measure each.
  Then we can gradually reduce it and flatten for an output of 3 different values, the highest of which is hopefully the correct answer.
  """
  def __init__(self):
    super().__init__()
    self.flatten = torch.nn.Flatten()
    self.linear_relu_stack = torch.nn.Sequential(
      torch.nn.Linear(63,32),
      torch.nn.ReLU(),
      torch.nn.Linear(32,32),
      torch.nn.ReLU(),
      torch.nn.Linear(32,16),
      torch.nn.ReLU(),
      torch.nn.Linear(16,3),
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits