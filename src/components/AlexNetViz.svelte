<script>
  import { onMount } from 'svelte';
  import * as d3 from 'd3';

  let svg;

  onMount(() => {
    const width = 1000;
    const height = 400; // Adjusted height to remove black space

    const layers = [
      { type: 'input', x: 50, y: 50, width: 100, height: 100, label: 'Input\n227x227x3' },
      { type: 'conv', x: 200, y: 50, width: 100, height: 100, label: 'CONV\n11x11,\nstride=4|\n96 kernels' },
      { type: 'pool', x: 350, y: 75, width: 100, height: 50, label: 'Overlapping\nMax POOL\n3x3,\nstride=2' },
      { type: 'conv', x: 500, y: 50, width: 100, height: 100, label: 'CONV\n5x5,pad=2| \n256 kernels' },
      { type: 'pool', x: 650, y: 75, width: 100, height: 50, label: 'Overlapping\nMax POOL\n3x3,\nstride=2' },
      { type: 'conv', x: 50, y: 200, width: 100, height: 100, label: 'CONV\n3x3,pad=1| \n384 kernels' },
      { type: 'conv', x: 200, y: 200, width: 100, height: 100, label: 'CONV\n3x3,pad=1| \n384 kernels' },
      { type: 'conv', x: 350, y: 200, width: 100, height: 100, label: 'CONV\n3x3,pad=1| \n256 kernels' },
      { type: 'pool', x: 500, y: 225, width: 100, height: 50, label: 'Overlapping\nMax POOL\n3x3,\nstride=2' },
      { type: 'fc', x: 650, y: 175, width: 50, height: 200, label: 'FC\n4096' },
      { type: 'fc', x: 750, y: 175, width: 50, height: 200, label: 'FC\n4096' },
      { type: 'softmax', x: 850, y: 225, width: 50, height: 100, label: '1000\nSoftmax' }
    ];

    svg = d3.select('#svg-container')
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // Define gradients
    const defs = svg.append('defs');

    const gradientColors = {
      'input': ['#0000ff', '#0000aa'],
      'conv': ['#ff8c00', '#ff4500'],
      'pool': ['#add8e6', '#87ceeb'],
      'fc': ['#ff8c00', '#ff4500'],
      'softmax': ['#008000', '#006400']
    };

    Object.keys(gradientColors).forEach(type => {
      const gradient = defs.append('linearGradient')
        .attr('id', `${type}-gradient`)
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '100%');

      gradient.append('stop')
        .attr('offset', '0%')
        .attr('style', `stop-color:${gradientColors[type][0]};stop-opacity:1`);

      gradient.append('stop')
        .attr('offset', '100%')
        .attr('style', `stop-color:${gradientColors[type][1]};stop-opacity:1`);
    });

    layers.forEach(layer => {
      if (layer.type === 'input' || layer.type === 'conv' || layer.type === 'pool' || layer.type === 'fc' || layer.type === 'softmax') {
        svg.append('rect')
          .attr('x', layer.x)
          .attr('y', layer.y)
          .attr('width', layer.width)
          .attr('height', layer.height)
          .attr('fill', `url(#${layer.type}-gradient)`)
          .attr('stroke', 'black');
      }

      svg.append('text')
        .attr('x', layer.x + layer.width / 2) // Center text horizontally
        .attr('y', layer.y + layer.height / 2) // Center text vertically
        .attr('dy', '.35em')
        .attr('text-anchor', 'middle')
        .style('font-size', '10px')
        .style('fill', 'black') // Change text color to black
        .text(layer.label)
        .call(wrap, layer.width - 10); // Apply text wrapping
    });

    // Add circles inside the FC and softmax layers
    const circleData = [
      { x: 675, y: 200, radius: 10 },
      { x: 675, y: 250, radius: 10 },
      { x: 675, y: 300, radius: 10 },
      { x: 675, y: 350, radius: 10 },
      { x: 775, y: 200, radius: 10 },
      { x: 775, y: 250, radius: 10 },
      { x: 775, y: 300, radius: 10 },
      { x: 775, y: 350, radius: 10 },
      { x: 875, y: 250, radius: 10 },
      { x: 875, y: 300, radius: 10 }
    ];

    circleData.forEach(circle => {
      svg.append('circle')
        .attr('cx', circle.x)
        .attr('cy', circle.y)
        .attr('r', circle.radius)
        .attr('fill', 'white')
        .attr('stroke', 'black');
    });

    // Add arrows and lines to connect the elements
    const arrowData = [
      { x1: 150, y1: 100, x2: 200, y2: 100 },
      { x1: 300, y1: 100, x2: 350, y2: 100 },
      { x1: 450, y1: 100, x2: 500, y2: 100 },
      { x1: 600, y1: 100, x2: 650, y2: 100 },
      { x1: 150, y1: 250, x2: 200, y2: 250 },
      { x1: 300, y1: 250, x2: 350, y2: 250 },
      { x1: 450, y1: 250, x2: 500, y2: 250 },
      { x1: 600, y1: 250, x2: 650, y2: 250 },
      { x1: 700, y1: 250, x2: 750, y2: 250 },
      { x1: 800, y1: 250, x2: 850, y2: 250 }
    ];

    arrowData.forEach(arrow => {
      svg.append('line')
        .attr('x1', arrow.x1)
        .attr('y1', arrow.y1)
        .attr('x2', arrow.x2)
        .attr('y2', arrow.y2)
        .attr('stroke', 'white')
        .attr('marker-end', 'url(#arrow)');
    });

    svg.append('defs').append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', '5')
      .attr('refY', '5')
      .attr('markerWidth', '6')
      .attr('markerHeight', '6')
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', 'white');
  });

  function wrap(text, width) {
    text.each(function() {
      const text = d3.select(this),
        words = text.text().split("\n").reverse(),
        lineHeight = 1.1, // ems
        y = text.attr("y"),
        dy = parseFloat(text.attr("dy"));

      let word,
        line = [],
        lineNumber = 0,
        tspan = text.text(null).append("tspan").attr("x", text.attr("x")).attr("y", y).attr("dy", dy + "em");

      while (word = words.pop()) {
        line.push(word);
        tspan.text(line.join(" "));
        if (tspan.node().getComputedTextLength() > width) {
          line.pop();
          tspan.text(line.join(" "));
          line = [word];
          tspan = text.append("tspan").attr("x", text.attr("x")).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
        }
      }
    });
  }
</script>

<div id="svg-container"></div>

<style>
  #svg-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: auto; /* Adjust height to auto to remove unnecessary space */
    background-color: black;
  }
  text {
    pointer-events: none;
  }
</style>