<script>
  export let show = false;
  export let onClose;
</script>

{#if show}
  <div class="popup">
      <div class="close" on:click={onClose}>✖</div>
      <h2>Hidden Layer 1 Explanation</h2>
      <h3>Code Representation</h3>
      <p>This is a simple CNN architecture. The code that is not green is where HL1 is relevant.</p>
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

                    # Fully connected hidden layers<span style="color: #D8BFD8;">
                    self.fc1 = nn.Linear(16*32*32, 128)  # First Hidden Layer
                    </span>self.fc2 = nn.Linear(128, 64)
                    # Output layer
                    self.out = nn.Linear(64, num_classes)
            
                def forward(self, x):
                   
                    # Apply convolutional layer
                    x = self.conv1(x)
                    x = F.relu(x)
                    x = F.max_pool2d(x, 2)
                   <span style="color: #D8BFD8;">
                    # Flatten the output from the convolutional layer
                    x = x.view(x.size(0), -1)
                    
                    # Apply fully connected hidden layer 1
                    x = self.fc1(x)
                    x = F.relu(x)
                  </span>
                    
                    x = self.fc2(x)
                    x = F.relu(x)
                    
                    # Apply output layer
                    x = self.out(x)
                    
                    return x
            
            # Example usage
            model = SimpleCNN(num_classes=10)
          </code>
      </pre>

      <h3>Mathematical Representation of Fully Connected Layer (fc1)</h3>
      <p>The fully connected layer can be defined as:</p>
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

      <h3>Matrix Transformation of Fully Connected Layer (fc1)</h3>
      <p>Example of transformation using input matrix [[6, 8], [12, 14]]:</p>
      <pre>
          <code>
              Example Input Matrix (x):
              6 8
              12 14

              Input Matrix After x = x.view(x.size(0), -1):
              6
              8
              12
              14

              Example Weight Matrix (W):
              1 0 0 0
              0 1 0 0
              0 0 1 0
              0 0 0 1

              Bias Vector (b):
              1
              1
              1
              1

              Transformation (y = Wx + b):
              o1 = [1, 0, 0, 0] * [6] + [1] = [6 + 1]   = 7
              o2 = [0, 1, 0, 0] * [8] + [1] = [8 + 1]   = 9
              o3 = [0, 0, 1, 0] * [12] + [1] = [12 + 1] = 13
              o4 = [0, 0, 0, 1] * [14] + [1] = [14 + 1] = 15
          </code>
      </pre>

      <h3>Image Transformation</h3>
        <div class="images-transformation">
            <div class="image-column">
                <img src="/6_CL.png" alt="Original 6" />
                <div class="arrow">↓</div>
                <img src="/6_HL1.png" alt="Transformed 6" />
            </div>
            <div class="image-column">
                <img src="/0_CL.png" alt="Original 0" />
                <div class="arrow">↓</div>
                <img src="/0_HL1.png" alt="Transformed 0" />
            </div>
        </div>
        <p>
          This output is the result of linearly transforming the output from the CL from 16x32x32 to 128. While the conversion 
          of the array into an image is unreadable to us, it's readable to our computer.
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

  .images-transformation {
    display: flex;
    justify-content: space-around;
    margin-top: 20px;
  }

  .image-column {
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .image-column img {
    max-width: 250px; /* Adjust this value to ensure images are not too large */
    margin-bottom: 10px;
    border: 1px solid white;
  }

  .arrow {
    font-size: 24px;
    color: white;
  }
</style>