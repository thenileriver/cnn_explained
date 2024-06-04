<head>
  <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap" rel="stylesheet">
</head>

<script lang="ts">
  import { onMount } from 'svelte';
  import Visualization from './cnn_viz.svelte';

  let selectedArchitecture = "AlexNet";

  const architectures = ["AlexNet", "VGG16", "ResNet", "Inception"];

  const architectureImages = {
    "AlexNet": "static/AlexNet_architecture.png",
    "VGG16": "static/VGG16_architecture.png",
    "ResNet": "static/ResNet_architecture.png",
    "Inception": "static/Inception_architecture.png"
  };

  function handleArchitectureChange(event) {
    selectedArchitecture = event.target.value;
  }
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
    max-width: 800px;
    margin: 0 auto;
    padding: 100px; /* Adjust padding as needed */
  }

  h1 {
    position: absolute;
    top: 20px;
    left: 20px;
    font-size: 2rem; /* Base font size for h1 */
    margin: 0; /* Update text color to white */
    font-weight: 300; /* Lighter font weight */
    display: flex;
    flex-wrap: wrap;
  }

  h1 span {
    display: inline-block;
    margin-right: 5px; /* Space between words */
  }

  h1 span::first-letter {
    font-size: 4.5rem; /* Adjust size as needed */
  }

  h2 {
    text-align: center;
    font-size: 2rem; /* Larger font size */
    margin-bottom: 10px; /* Reduced margin */
    font-weight: 300; /* Lighter font weight */
    margin-top: 20px; /* Move h2 lower to prevent interference with h1 */
  }

  .section {
    padding: 20px;
    margin: 20px 0;
    background-color: #121212; /* Dark gray background */
    border: none; /* Remove border */
    border-radius: 10px;
    transition: opacity 1s ease-out, transform 1s ease-out;
  }

  .hidden {
    opacity: 0;
    transform: translateY(100px);
  }

  .visible {
    opacity: 1;
    transform: translateY(0);
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
    height: 600px; /* Adjust the height as needed */
    font-size: 1.2rem;
    flex-direction: column;
  }

  .visualization img {
    max-width: 100%;
    max-height: 100%;
  }
</style>

<div class="container">
  <h1>
    <span>Convolutional</span>
    <span>Neural</span>
    <span>Network</span>
  </h1>
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
      <p>Selected Architecture: {selectedArchitecture}</p>
      {#if architectureImages[selectedArchitecture]}
        <img src={architectureImages[selectedArchitecture]} alt={selectedArchitecture} />
      {/if}
    </div>
  </div>
</div>