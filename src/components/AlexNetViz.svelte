<script>
    import { onMount } from 'svelte';
    import * as d3 from 'd3';
  
    let svg;
  
    onMount(() => {
      const width = 1200;
      const height = 400;
  
      const layers = [
        { type: 'conv', x: 10, y: 50, width: 100, height: 100, label: 'Conv1\n96x11x11\nstride=4\npad=0' },
        { type: 'pool', x: 120, y: 75, width: 100, height: 50, label: 'Max Pool\n3x3\nstride=2\npad=0' },
        { type: 'conv', x: 230, y: 50, width: 100, height: 100, label: 'Conv2\n256x5x5\nstride=1\npad=2' },
        { type: 'pool', x: 340, y: 75, width: 100, height: 50, label: 'Max Pool\n3x3\nstride=2\npad=0' },
        { type: 'conv', x: 450, y: 50, width: 100, height: 100, label: 'Conv3\n384x3x3\nstride=1\npad=1' },
        { type: 'conv', x: 560, y: 50, width: 100, height: 100, label: 'Conv4\n384x3x3\nstride=1\npad=1' },
        { type: 'conv', x: 670, y: 50, width: 100, height: 100, label: 'Conv5\n256x3x3\nstride=1\npad=1' },
        { type: 'pool', x: 780, y: 75, width: 100, height: 50, label: 'Max Pool\n3x3\nstride=2\npad=0' },
        { type: 'fc', x: 890, y: 25, width: 50, height: 150, label: 'FC\n4096' },
        { type: 'fc', x: 950, y: 25, width: 50, height: 150, label: 'FC\n4096' },
        { type: 'softmax', x: 1010, y: 25, width: 50, height: 150, label: 'Softmax\n1000' }
      ];
  
      const svgElement = d3.select(svg)
        .attr('width', width)
        .attr('height', height);
  
      const layerGroup = svgElement.selectAll('g')
        .data(layers)
        .enter()
        .append('g')
        .attr('transform', d => `translate(${d.x}, ${d.y})`);
  
      layerGroup.append('rect')
        .attr('width', d => d.width)
        .attr('height', d => d.height)
        .attr('fill', d => d.type === 'conv' ? 'orange' : d.type === 'pool' ? 'lightblue' : d.type === 'fc' ? 'green' : 'purple')
        .attr('stroke', '#000')
        .attr('stroke-width', 2);
  
      layerGroup.append('text')
        .attr('x', 5)
        .attr('y', 20)
        .attr('fill', 'white')
        .attr('font-size', '10px')
        .attr('dy', '.35em')
        .text(d => d.label)
        .call(wrap, 90); // Wrap text within 90px width
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
          tspan = text.text(null).append("tspan").attr("x", 5).attr("y", y).attr("dy", dy + "em");
  
        while (word = words.pop()) {
          line.push(word);
          tspan.text(line.join(" "));
          if (tspan.node().getComputedTextLength() > width) {
            line.pop();
            tspan.text(line.join(" "));
            line = [word];
            tspan = text.append("tspan").attr("x", 5).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
          }
        }
      });
    }
  </script>
  
  <svg bind:this={svg}></svg>
  
  <style>
    svg {
      border: 1px solid #ccc;
    }
    text {
      pointer-events: none;
    }
  </style>