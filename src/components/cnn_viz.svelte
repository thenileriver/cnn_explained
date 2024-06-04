<script lang="ts">
  import Slideshow from './slideshow.svelte';
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  const layers = [
    { id: 'input_image', nodes: 1, label: 'Input Image' },
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

    for (let i = 0; i < layerCenters.length - 1; i++) {
      const currentLayer = layerCenters[i];
      const nextLayer = layerCenters[i + 1];

      for (let j = 0; j < currentLayer.nodes; j++) {
        for (let k = 0; k < nextLayer.nodes; k++) {
          svg.append('line')
            .attr('x1', currentLayer.x + 20)
            .attr('y1', currentLayer.y + j * nodeSpacing + nodeSpacing / 4)
            .attr('x2', nextLayer.x)
            .attr('y2', nextLayer.y + k * nodeSpacing + nodeSpacing / 4)
            .attr('stroke', 'lightgray')
            .attr('stroke-width', 1)
            .classed(`line-from-${i}-${j}-to-${i+1}-${k}`, true);
        }
      }
    }

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

          d3.selectAll('circle')
            .classed('clicked', false)
            .style('fill', '#00A1E4');

          d3.selectAll('line')
            .classed('highlighted', false)
            .style('stroke', 'lightgray')
            .style('stroke-width', 1);

          circle.classed('clicked', true)
                .style('fill', '#00ffff');

          const nodeIndex = d;
          const layerIndex = i;

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

          circle.classed('clicked', false)
                .style('fill', '#00A1E4');

          const nodeIndex = d;
          const layerIndex = i;

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

<div class="visualization-container">
  <Slideshow />
  <div class="visualization">
    <svg id="cnn-vis" width="800" height="600"></svg>
  </div>
</div>
