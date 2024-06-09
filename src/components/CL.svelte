<script>
    export let show = false;
    export let onClose;
  </script>
  
  {#if show}
    <div class="popup">
        <div class="close" on:click={onClose}>✖</div>
        <h2>Convolutional Layer Explanation</h2>
        <h3>Code Representation</h3>
        <pre>
            <code>
              import torch
              import torch.nn as nn
              import torch.nn.functional as F
              
              class SimpleCNN(nn.Module):
                  def __init__(self, num_classes=10):
                      super(SimpleCNN, self).__init__()
                      <span style="color: #D8BFD8;">
                      self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
                      </span>
                      # Fully connected hidden layers
                      self.fc1 = nn.Linear(16*32*32, 128)  # Adjust the input size based on the output size of the conv layer
                      self.fc2 = nn.Linear(128, 64)
                      
                      # Output layer
                      self.out = nn.Linear(64, num_classes)
              
                  def forward(self, x):
                     <span style="color: #D8BFD8;">
                      # Apply convolutional layer
                      x = self.conv1(x)
                      x = F.relu(x)
                      x = F.max_pool2d(x, 2)
                     </span>
                      # Flatten the output from the convolutional layer
                      x = x.view(x.size(0), -1)
                      
                      # Apply fully connected hidden layers
                      x = self.fc1(x)
                      x = F.relu(x)
                      
                      x = self.fc2(x)
                      x = F.relu(x)
                      
                      # Apply output layer
                      x = self.out(x)
                      
                      return x
              
              # Example usage
              model = SimpleCNN(num_classes=10)
            </code>
        </pre>
  
        <h3>Mathematical Representation</h3>
        <p>The convolution operation can be defined as:</p>
        <pre>
            <code>
                (I * K)(i, j) = Σ Σ I(m, n) * K(i-m, j-n)
            </code>
        </pre>
  
        <h3>Legend</h3>
        <ul>
            <li><strong>I:</strong> Input matrix</li>
            <li><strong>K:</strong> Kernel matrix</li>
            <li><strong>(i, j):</strong> Coordinates of the output matrix</li>
            <li><strong>(m, n):</strong> Coordinates of the kernel matrix</li>
        </ul>
  
        <h3>Matrix Transformation</h3>
        <p>Example of a 3x3 input matrix and a 2x2 kernel:</p>
        <pre>
            <code>
                Input Matrix (I):
                1 2 3
                4 5 6
                7 8 9
                
                Kernel (K):
                1 0
                0 1
                
                Output Matrix (O):
                (I * K)(1, 1) = 1*1 + 2*0 + 4*0 + 5*1 = 1 + 5 = 6
                (I * K)(1, 2) = 2*1 + 3*0 + 5*0 + 6*1 = 2 + 6 = 8
                (I * K)(2, 1) = 4*1 + 5*0 + 7*0 + 8*1 = 4 + 8 = 12
                (I * K)(2, 2) = 5*1 + 6*0 + 8*0 + 9*1 = 5 + 9 = 14
                
                Result:
                6 8
                12 14
            </code>
        </pre>
        <p>This hypothetical 3x3 matrix would be a portion of the overall image. We run through the entire image in 3x3 portions, and transform that portion into 2x2 portions.
            By doing so, we reduce the overall complexity of the image, allowing the model to capture distinct features.
        </p>
    </div>
  {/if}
  
  <style>
    .popup {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: black;
        color: white;
        padding: 20px;
        border: 1px solid white;
        z-index: 10;
        width: 800px;
        height: 600px;
        overflow-y: auto;
    }
  
    .close {
        position: absolute;
        top: 10px;
        right: 10px;
        cursor: pointer;
        color: red;
        border: 1px dotted red;
        padding: 2px 5px;
        font-size: 18px;
        font-weight: bold;
    }
  
    pre {
        background-color: #333;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
  
    code {
        color: #00ff00;
    }
  </style>  