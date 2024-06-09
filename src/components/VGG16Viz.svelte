<script>
    // Add any necessary imports and script logic here
  </script>
  
  <style>
    /* Add your styles here */
    .vgg16-viz {
      display: flex;
      flex-direction: column;
      align-items: center;
      max-width: 100%; /* Ensure it doesn't overflow the parent container */
      margin: 0 auto; /* Center the visualization */
    }
    .row {
      display: flex;
      align-items: center;
      margin-bottom: 20px;
    }
    .layer {
      display: flex;
      flex-direction: row;
      align-items: center;
      justify-content: center;
      margin: 5px;
    }
    .block {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin: 2px;
    }
    .conv, .pool, .fc, .dropout, .softmax, .input {
      border: 1px solid black;
      padding: 5px;
      margin: 5px;
      width: 40px; /* Thinner width */
      height: 100px; /* Adjusted length */
    }
    .conv {
      background: linear-gradient(to bottom, #f28500, #e55d00); /* orange gradient */
    }
    .pool {
      background: linear-gradient(to bottom, #0000ff, #000099); /* blue gradient */
    }
    .fc {
      background: linear-gradient(to bottom, #ffd700, #e6c200); /* gold gradient */
      width: 10px; /* Thin width */
      height: 85px; /* Original height */
    }
    .dropout {
      background: linear-gradient(to bottom, #ff0000, #cc0000); /* red gradient */
      width: 10px; /* Thin width */
      height: 85px; /* Original height */
    }
    .softmax {
      background: linear-gradient(to bottom, #00ff00, #009900); /* green gradient */
      width: 25px; /* Square width */
      height: 25px; /* Square height */
    }
    .input {
      background-color: purple;
    }
    .arrow {
      width: 50px;
      height: 50px;
    }
    .arrow line, .arrow polygon {
      stroke: white;
      fill: white;
    }
    .tallest {
      height: 160px; /* Tallest height for stages 1 and 2 */
    }
    .label {
      font-size: 10px;
      font-family: Arial, sans-serif; /* Same font as AlexNetViz.svelte */
      text-align: center;
      margin-top: 5px;
    }
  </style>
  
  <div class="vgg16-viz">
    <div class="row">
      <div class="block">
        <div class="label">input<br>224x224x3</div>
        <div class="input tallest"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">conv1<br>224x224x64</div>
        <div class="conv tallest"></div>
      </div>
      <div class="block">
        <div class="label">conv2<br>224x224x64</div>
        <div class="conv tallest"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">pool1<br>112x112x64</div>
        <div class="pool" style="height:136px;"></div>
      </div>
      <div class="block">
        <div class="label">conv3<br>112x112x128</div>
        <div class="conv" style="height:136px;"></div>
      </div>
      <div class="block">
        <div class="label">conv4<br>112x112x128</div>
        <div class="conv" style="height:136px;"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">pool2<br>56x56x128</div>
        <div class="pool" style="height:100px;"></div>
      </div>
      <div class="block">
        <div class="label">conv5<br>56x56x256</div>
        <div class="conv" style="height:100px;"></div>
      </div>
      <div class="block">
        <div class="label">conv6<br>56x56x256</div>
        <div class="conv" style="height:100px;"></div>
      </div>
      <div class="block">
        <div class="label">conv7<br>56x56x256</div>
        <div class="conv" style="height:100px;"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
    </div>
    <div class="row">
      <div class="block">
        <div class="label">pool3<br>28x28x256</div>
        <div class="pool" style="height:80px;"></div>
      </div>
      <div class="block">
        <div class="label">conv8<br>28x28x512</div>
        <div class="conv" style="height:80px;"></div>
      </div>
      <div class="block">
        <div class="label">conv9<br>28x28x512</div>
        <div class="conv" style="height:80px;"></div>
      </div>
      <div class="block">
        <div class="label">conv10<br>28x28x512</div>
        <div class="conv" style="height:80px;"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">pool4<br>14x14x512</div>
        <div class="pool" style="height:60px;"></div>
      </div>
      <div class="block">
        <div class="label">conv11<br>14x14x512</div>
        <div class="conv" style="height:60px;"></div>
      </div>
      <div class="block">
        <div class="label">conv12<br>14x14x512</div>
        <div class="conv" style="height:60px;"></div>
      </div>
      <div class="block">
        <div class="label">conv13<br>14x14x512</div>
        <div class="conv" style="height:60px;"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">pool5<br>7x7x512</div>
        <div class="pool" style="height:37px;"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">fc1<br>1x1x4096</div>
        <div class="fc"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">dropout1</div>
        <div class="dropout"></div>
      </div>
      <svg class="arrow">
        <line x1="0" y1="25" x2="50" y2="25" />
        <polygon points="50,25 40,20 40,30" />
      </svg>
      <div class="block">
        <div class="label">softmax<br>1x1x1000</div>
        <div class="softmax"></div>
      </div>
    </div>
  </div>
