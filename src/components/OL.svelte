<script>
  export let show = false;
  export let onClose;
</script>

{#if show}
  <div class="popup">
      <div class="close" on:click={onClose}>✖</div>
      <h2>Output Layer Explanation</h2>
      <h3>Code Representation</h3>
      <pre>
          <code>
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            class SimpleCNN(nn.Module):
                def __init__(self, num_classes=10):
                    super(SimpleCNN, self).__init__()

                    #Convolutional Layer
                    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

                    # Fully connected hidden layers
                    self.fc1 = nn.Linear(16*32*32, 128)
                    self.fc2 = nn.Linear(128, 64)         # Second Hidden Layer

                    # Output layer<span style="color: #D8BFD8;">
                    self.out = nn.Linear(64, num_classes)</span>
            
                def forward(self, x):
                   
                    # Apply convolutional layer
                    x = self.conv1(x)
                    x = F.relu(x)
                    x = F.max_pool2d(x, 2)

                    # Flatten the output from the convolutional layer
                    x = x.view(x.size(0), -1)
                    
                    # Apply fully connected hidden layer 1
                    x = self.fc1(x)
                    x = F.relu(x)
                  
                    x = self.fc2(x)
                    x = F.relu(x)
                  <span style="color: #D8BFD8;">
                    # Apply output layer
                    x = self.out(x)</span>
                    
                    return x
            
            # Example usage
            model = SimpleCNN(num_classes=10)
          </code>
      </pre>

      <h3>Mathematical Representation of Output Layer</h3>
      <p>The output layer can be defined as:</p>
      <pre>
          <code>
              y = Wx + b
          </code>
      </pre>
      <h3>Legend</h3>
      <ul>
        <li><strong>y:</strong> Output vector</li>
        <li><strong>W:</strong> Weight matrix</li>
        <li><strong>x:</strong> Input vector</li>
        <li><strong>b:</strong> Bias vector</li>
      </ul>

      <h3>Matrix Transformation of Output Layer</h3>
      <p>Example of transformation using input matrix [[8], [10]]:</p>
      <pre>
        <code>
            Input Matrix (x):
            8
            10

            Weight Matrix (W):
            1 0
            0 1

            Bias Vector (b):
            1
            1

            Transformation (y = Wx + b):
            o1 = [1, 0] * [8] + [1]  = 9
            02 = [0, 1] * [10] + [1] = 11
        </code>
    </pre>
    <p>Assume 9 is the value for class 0 and 11 is the value for class 1. In this case, since 11 is greater than 9, the model would select class 1.</p>
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