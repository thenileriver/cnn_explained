<script lang="ts">
  import { onMount } from 'svelte';
  import Visualization from './cnn_viz.svelte';
  import AlexNetViz from './AlexNetViz.svelte'; // Import the AlexNetViz component
  import VGG16Viz from './VGG16Viz.svelte'; // Import the VGG16Viz component

  let selectedArchitecture = "AlexNet";

  const architectures = ["AlexNet", "VGG16"];

  const architectureImages = {
    "AlexNet": "AlexNet_architecture.png",
    "VGG16": "VGG16_architecture.png"
  };

  function handleArchitectureChange(event) {
    selectedArchitecture = event.target.value;
  }

  onMount(() => {
    const script = document.createElement('script');
    script.src = 'https://polyfill.io/v3/polyfill.min.js?features=es6';
    script.async = true;
    document.head.appendChild(script);

    const mathJaxScript = document.createElement('script');
    mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    mathJaxScript.async = true;
    document.head.appendChild(mathJaxScript);
  });
</script>

<style>
  :global(body) {
    margin: 0;
    font-family: 'Source Sans Pro', sans-serif;
    background-color: #121212; /* Update background to black */
    color: #f6f6f6; /* Update text color to white */
    overflow-x: hidden;
  }

  .container {
    max-width: 1000px; /* Adjusted maximum width */
    margin: 0 auto;
    padding: 20px; /* Adjusted padding */
  }

  h1 {
    font-size: 2.5rem; /* Base font size for h1 */
    margin: 0; /* Update text color to white */
    font-weight: 600; /* Adjusted font weight */
    text-align: center; /* Center-align the title */
    margin-bottom: 30px; /* Space between title and content */
  }

  h2 {
    text-align: center;
    font-size: 1.5rem; /* Adjusted font size */
    margin: 20px 0; /* Space around h2 */
    font-weight: 600; /* Adjusted font weight */
  }

  .section {
    padding: 20px;
    margin: 20px 0; /* Add margin between sections */
    background-color: #1e1e1e; /* Dark gray background */
    border-radius: 10px;
  }

  .architecture-select {
    display: flex;
    justify-content: center;
    margin: 20px 0;
  }

  .architecture-select select {
    padding: 10px;
    background-color: #FFFFFF;
    color: #333;
    border: 1px solid #E0E0E0;
    border-radius: 5px;
    font-size: 1rem;
  }

  .visualization {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    height: auto; /* Allow height to adjust based on content */
    width: 100%;
    font-size: 1.2rem;
    margin-top: 20px; /* Add space between architecture text and visualization */
  }

  .visualization img {
    max-width: 100%;
    height: auto; /* Maintain aspect ratio */
  }

  .cool-title {
    font-family: 'Arial', sans-serif;
    font-size: 3em;
    color: #9d0c9d; /* Cool purple color */
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    margin-top: 50px;
    letter-spacing: 2px;
  }

  .text-area {
    margin: 20px 0; /* Space around text area */
    background-color: #1e1e1e; /* Dark gray background */
    padding: 20px;
    border-radius: 10px;
  }

  .text-area p {
    margin: 0;
    white-space: pre-line; /* Collapse whitespace while preserving line breaks */
  }
</style>

<div class="container">
  <h1 class="cool-title">Computer Vision and Convolutional Neural Network Visualization</h1>

  <div class="section text-area">
    <p>
      From simple object detection to self-driving cars, the field of computer vision holds a range of industry applications and untapped potential, and it's all powered by one type of neural network, convolutional neural networks, also known as CNNs.

      CNNs, as the name suggests, are powered by a mathematical formula called convolutions represented as:
      <span id="math-formula">(f * g)(t) = ∫<sub>−∞</sub><sup>∞</sup> f(τ) g(t − τ) dτ</span>

      Convolutions are essentially a way of combining two functions to produce a third function that expresses how the shape of one is modified by the other. In the context of CNNs, the first function is typically the input image, and the second is a filter or kernel. This process involves sliding the filter over the image and computing the dot product between the filter and the overlapping image region.

      The key advantage of CNNs lies in their ability to automatically and adaptively learn spatial hierarchies of features from input images. This is achieved through a series of layers, each designed to perform specific operations:

      <ul>
        <li><strong>Convolutional Layer:</strong> This layer applies a set of convolutional filters to the input image, extracting various features such as edges, textures, and patterns. The result is a set of feature maps, each corresponding to different features.</li>
        <li><strong>Activation Function (ReLU):</strong> After convolution, an activation function like ReLU (Rectified Linear Unit) is applied to introduce non-linearity into the model, enabling it to learn complex patterns.</li>
        <li><strong>Pooling Layer:</strong> This layer reduces the spatial dimensions of the feature maps, typically using operations like max pooling. This helps in reducing the computational load and controlling overfitting.</li>
        <li><strong>Fully Connected Layer:</strong> In the final stages, the network includes one or more fully connected layers where the high-level reasoning takes place. The output of these layers is usually a class score, representing the likelihood of various possible outcomes.</li>
      </ul>

      <br>
      CNNs work best when images are as simple as possible, so a key part of a computer vision model's pipeline is the transformation of the images before the images are fed into the CNN.

      <br><br><br><br>
      Below, you can interact with both the slideshow and the visualization of a simple CNN architecture. The slideshow goes over some basic image transformations and the visualization documents the journey an image goes through as it passes through the layers of a CNN.
  </div>

  <div class="section">
    <Visualization />
  </div>

  <div class="section">
    <h2>Select CNN Architecture</h2>
    <div class="architecture-select">
      <select on:change={handleArchitectureChange}>
        {#each architectures as architecture}
          <option value={architecture}>{architecture}</option>
        {/each}
      </select>
    </div>
    <div class="visualization">
      <p>Selected Architecture: {selectedArchitecture}</p> <!-- Moved this line above the visualization -->
      {#if selectedArchitecture === 'AlexNet'}
        <AlexNetViz /> <!-- Display AlexNetViz component for AlexNet -->
        <p>The AlexNet architecture excels at image classification tasks, here's a breakdown on how this architecture makes it one of the best<br><br>

          Input Layer: Starting point where image is fed into the network as a 227x227x3 RGB image<br><br>

          Convolutional Layer 1: With kernel size 11x11, stride 4 and 96 kernels, we start off with a larger kernel to help capture low-level features like edges and textures<br><br>

          Overlapping Max Pooling 1: With pooling size 3x3 and strid 4, this layer reduces the dimensionality of the arrays<br><br>

          Convolutional Layer 2: With kernel size 5x5, paddng 2, and 256 kernels, this layer extracts more features<br><br>

          Overlapping Max Pooling 2: With pooiling size 3x3 and stride 2, we further reduce the dimensionality while maintaining important features<br><br>

          Convolutional Layers 3, 4, 5: All 3 have kernel size 3x3 and padding 1, layer 3 and 4 have 384 kernels while layer 5 has 256 kernels, these 3 layers refine features and increase the depth of the neural network, allowing it to learning more complex features<br><br>

          Overlapping Max Pooling 3: With pooling size 3x4 and stride 2, this final pooling layer reduces dimensionality before being fed into the fully connected layers<br><br>

          Fully Connected Layers (FC6 and FC7): With both layers having 4096 neurons, these layers help learn high-level features that help drive decision making in the model<br><br>

          Output Layer FC8: 1000 neurons corresponding to the 1000 classes in the ImageNet dataset which this mode was trained on with Softmax activation, this layer provides the final predictions<br><br>
        </p>
      {:else if selectedArchitecture === 'VGG16'}
        <VGG16Viz />
        <p>
          The VGG16 architecture was developed at Oxford University, and excels at its accracy to computational cost ratio. Despite taking more simpler approaches, it achieves similar accuracy to more "complex" models.
        </p>
      {/if}
    </div>
  </div>
</div>