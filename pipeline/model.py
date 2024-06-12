from torch import nn

class InputLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(InputLayer, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
  
  def forward(self, x):
    return self.linear(x)
    
        
class OutputLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(OutputLayer, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    
  def forward(self, x):
      return self.linear(x)
