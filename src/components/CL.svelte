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
              
              class ConvLayer(nn.Module):
                  def __init__(self):
                      super(ConvLayer, self).__init__()
                      self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
                  
                  def forward(self, x):
                      x = self.conv1(x)
                      return x
              
              # Example usage:
              layer = ConvLayer()
              input_tensor = torch.randn(1, 1, 5, 5)  # Batch size 1, 1 channel, 5x5 image
              output_tensor = layer(input_tensor)
              print(output_tensor)
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