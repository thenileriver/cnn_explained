<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  let selectedArchitecture = "AlexNet";

  const architectures = ["AlexNet", "VGG16", "ResNet", "Inception"];

  function handleArchitectureChange(event) {
      selectedArchitecture = event.target.value;
  }

  const layers = [
    { id: 'input', nodes: 4, label: 'Input Layer' },
    { id: 'hidden1', nodes: 6, label: 'Hidden Layer 1' },
    { id: 'hidden2', nodes: 6, label: 'Hidden Layer 2' },
    { id: 'output', nodes: 2, label: 'Output Layer' }
  ];

  onMount(() => {
    const svg = d3.select('#cnn-vis');
    const layerWidth = 150;
    const nodeSpacing = 40;
    const maxNodes = Math.max(...layers.map(layer => layer.nodes));

    const layerCenters = layers.map((layer, i) => {
      const yOffset = (maxNodes - layer.nodes) * nodeSpacing / 2;
      return {
        x: i * layerWidth + 100,
        y: 100 + yOffset,
        nodes: layer.nodes
      };
    });

    // Draw lines between layers
    for (let i = 0; i < layerCenters.length - 1; i++) {
      const currentLayer = layerCenters[i];
      const nextLayer = layerCenters[i + 1];

      for (let j = 0; j < currentLayer.nodes; j++) {
        for (let k = 0; k < nextLayer.nodes; k++) {
          svg.append('line')
            .attr('x1', currentLayer.x)
            .attr('y1', currentLayer.y + j * nodeSpacing)
            .attr('x2', nextLayer.x)
            .attr('y2', nextLayer.y + k * nodeSpacing)
            .attr('stroke', 'lightgray')
            .attr('stroke-width', 1)
            .classed(`line-from-${i}-${j}-to-${i+1}-${k}`, true);
        }
      }
    }

    // Draw nodes and labels
    layers.forEach((layer, i) => {
      const group = svg.append('g')
        .attr('transform', `translate(${layerCenters[i].x}, ${layerCenters[i].y})`);

      group.selectAll('circle')
        .data(d3.range(layer.nodes))
        .enter()
        .append('circle')
        .attr('cx', 0)
        .attr('cy', (d, index) => index * nodeSpacing)
        .attr('r', 10)
        .style('fill', 'steelblue')
        .attr('cursor', 'pointer')
        .on('mouseover', function(event, d) {
          const circle = d3.select(this);

          // Reset all nodes to blue
          d3.selectAll('circle')
            .classed('clicked', false)
            .style('fill', 'steelblue');

          // Reset all lines to not highlighted
          d3.selectAll('line')
            .classed('highlighted', false)
            .style('stroke', 'lightgray')
            .style('stroke-width', 1);

          // Set the hovered node to yellow
          circle.classed('clicked', true)
                .style('fill', '#FFD700'); // Darker yellow

          const nodeIndex = d;
          const layerIndex = i;

          // Highlight all outgoing lines connected to this node
          if (layerIndex < layers.length - 1) {
            for (let k = 0; k < layers[layerIndex + 1].nodes; k++) {
              d3.selectAll(`.line-from-${layerIndex}-${nodeIndex}-to-${layerIndex+1}-${k}`)
                .classed('highlighted', true)
                .style('stroke', '#FFD700')
                .style('stroke-width', 2);
            }
          }
        })
        .on('mouseout', function(event, d) {
          const circle = d3.select(this);

          // Reset the hovered node to blue
          circle.classed('clicked', false)
                .style('fill', 'steelblue');

          const nodeIndex = d;
          const layerIndex = i;

          // Reset all outgoing lines connected to this node to default color
          if (layerIndex < layers.length - 1) {
            for (let k = 0; k < layers[layerIndex + 1].nodes; k++) {
              d3.selectAll(`.line-from-${layerIndex}-${nodeIndex}-to-${layerIndex+1}-${k}`)
                .classed('highlighted', false)
                .style('stroke', 'lightgray')
                .style('stroke-width', 1);
            }
          }
        });

      group.append('text')
        .attr('x', 0)
        .attr('y', -20)
        .text(layer.label)
        .attr('text-anchor', 'middle');
    });
  });
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
    display: flex;
    justify-content: center;
    align-items: center;
    height: 600px; /* Adjust the height as needed */
    font-size: 1.2rem;
  }

  line {
    stroke: lightgray;
    stroke-width: 1px;
  }

  line.highlighted {
    stroke: #FFD700;
    stroke-width: 2px;
  }

  circle {
    stroke: #fff;
    stroke-width: 2px;
  }

  circle.clicked {
    fill: #FFD700;
  }

  text {
    font-size: 14px;
    font-family: 'Arial', sans-serif;
    font-weight: bold;
    fill: #333;
    pointer-events: none;
  }

</style>

<div class="container">
  <h1>Convolutional Neural Network Visualization</h1>

  <div class="section">
      <h2>Basic Neural Network Visualization</h2>
      <div class="visualization">
          <svg id="cnn-vis" width="800" height="600"></svg>
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
