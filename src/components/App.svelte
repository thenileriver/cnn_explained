<script>
  let selectedArchitecture = "AlexNet";

  const architectures = ["AlexNet", "VGG16", "ResNet", "Inception"];

  function handleArchitectureChange(event) {
      selectedArchitecture = event.target.value;
  }
</script>

<style>
  :global(body) {
      margin: 0;
      font-family: 'Arial', sans-serif;
      background-color: #121212;
      color: #ffffff;
      overflow-x: hidden;
  }

  .container {
      max-width: 800px;
      margin: 0 auto;
      padding: 20px;
  }

  h1 {
      text-align: center;
      font-size: 2.5rem;
      margin-bottom: 40px;
      color: #ffcc00;
      text-shadow: 2px 2px 4px #000000;
  }

  h2 {
      text-align: center;
      font-size: 2rem;
      margin-bottom: 20px;
      color: #ffcc00;
      text-shadow: 2px 2px 4px #000000;
  }

  .section {
      padding: 20px;
      margin: 20px 0;
      background-color: #1e1e1e;
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
      background-color: #333;
      color: #fff;
      border: 2px solid #ffcc00;
      border-radius: 5px;
      font-size: 1rem;
  }

  .visualization {
      text-align: center;
      font-size: 1.2rem;
  }
</style>

<div class="container">
  <h1>Convolutional Neural Network Visualization</h1>

  <div class="section">
      <h2>Basic Neural Network Visualization</h2>
      <div class="visualization">
          <!-- Placeholder for Basic Neural Network Visualization -->
          <p>[Basic Neural Network Visualization Here]</p>
      </div>
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
          <!-- Placeholder for CNN Architecture Visualization -->
          <p>[CNN Visualization for {selectedArchitecture} Here]</p>
      </div>
  </div>
</div>
