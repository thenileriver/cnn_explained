<head>
  <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap" rel="stylesheet">
</head>

<script lang="ts">
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  let selectedArchitecture = "AlexNet";

  const architectures = ["AlexNet", "VGG16", "ResNet", "Inception"];

  function handleArchitectureChange(event) {
      selectedArchitecture = event.target.value;
  }

  // Update the layers array to include the new layer
  const layers = [
    { id: 'input_image', nodes: 1, label: 'Input Image' }, // New layer with one node
    { id: 'input', nodes: 4, label: 'Convolution Layer' },
    { id: 'hidden1', nodes: 6, label: 'Hidden Layer 1' },
    { id: 'hidden2', nodes: 6, label: 'Hidden Layer 2' },
    { id: 'output', nodes: 2, label: 'Output Layer' }
  ];

  let svgWidth = 800;
  let svgHeight = 600;

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
            .attr('x1', currentLayer.x + 20) // Adjusted to start from the middle of the right side of the square
            .attr('y1', currentLayer.y + j * nodeSpacing + nodeSpacing / 4) // Adjusted to start from the middle of the right side of the square
            .attr('x2', nextLayer.x) // Adjusted to end at the middle of the left side of the square
            .attr('y2', nextLayer.y + k * nodeSpacing + nodeSpacing / 4) // Adjusted to end at the middle of the left side of the square
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

      group.selectAll('g')
        .data(d3.range(layer.nodes))
        .enter()
        .append('g')
        .attr('transform', (d, index) => `translate(0, ${index * nodeSpacing})`)
        .append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 20)
        .attr('height', 20)
        .attr('fill', '#00A1E4')
        .attr('cursor', 'pointer')
        .on('mouseover', function(event, d) {
          const circle = d3.select(this);

          // Reset all nodes to blue
          d3.selectAll('circle')
            .classed('clicked', false)
            .style('fill', '#00A1E4');

          // Reset all lines to not highlighted
          d3.selectAll('line')
            .classed('highlighted', false)
            .style('stroke', 'lightgray')
            .style('stroke-width', 1);

          // Set the hovered node to yellow
          circle.classed('clicked', true)
                .style('fill', '#00ffff');

          const nodeIndex = d;
          const layerIndex = i;

          // Highlight all outgoing and incoming lines connected to this node

          if (layerIndex > 0) {
            for (let k = 0; k < layers[layerIndex - 1].nodes; k++) {
              d3.selectAll(`.line-from-${layerIndex-1}-${k}-to-${layerIndex}-${nodeIndex}`)
                .classed('highlighted', true)
                .style('stroke', '#00ffff')
                .style('stroke-width', 2);
            }
          }
        })
        .on('mouseout', function(event, d) {
          const circle = d3.select(this);

          // Reset the hovered node to blue
          circle.classed('clicked', false)
                .style('fill', '#00A1E4');

          const nodeIndex = d;
          const layerIndex = i;

          // Reset all highlighted edges when mouse is not hovering
          d3.selectAll('.highlighted')
            .classed('highlighted', false)
            .style('stroke', 'lightgray')
            .style('stroke-width', 1);
        })
        .append('image')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 20)
        .attr('height', 20)
        .attr('href', 'path_to_node_image.jpg');

      group.append('text')
        .attr('x', 0)
        .attr('y', -20)
        .text(layer.label)
        .style('fill', 'white')
        .attr('text-anchor', 'middle');
    });
  });
</script>

<style>
:global(body) {
  margin: 0;
  font-family: 'Source Sans Pro', sans-serif;
  background-color: #121212; /* Update background to black */
  color: #f6f6f6;/* Update text color to white */
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
  margin: 0;/* Update text color to white */
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
}

line {
  stroke: none; /* Remove border */
}

circle {
  stroke: none; /* Remove border */
}

text {
  font-size: 14px;
  font-family: 'Source Sans Pro', sans-serif;
  font-weight: 600; /* Semi-bold text */
  fill: white; /* Update text color to white */
  pointer-events: none;
}

</style>

<div class="container">
  <h1>
    <span>Convolutional</span><br>
    <span>Neural</span><br>
    <span>Network</span><br>
  </h1>
  <div class="section">
    <div class="visualization">
      <svg id="cnn-vis" style="width: {svgWidth}px; height: {svgHeight}px">
      </svg>
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
      <p>[CNN Visualization for {selectedArchitecture} Here]</p>
    </div>
  </div>
</div>

