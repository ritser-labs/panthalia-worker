from torch import nn

class InputLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(InputLayer, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    
        
class OutputLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(OutputLayer, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
