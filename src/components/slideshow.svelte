<script lang="ts">
    import { onMount } from 'svelte';
  
    let currentSlideIndex = 0;
  
    const slides = [
      {
        image: "President_Barack_Obama.jpg", // Update with the correct path
        text: "This is the original image that will be processed."
      },
      {
        image: "/path/to/processed/image1.png", // Update with the correct path
        text: "Step 1: Image after initial processing."
      },
      {
        image: "path", // Update with the correct path
        text: "Step 2: Image after further processing."
      },
      // Add more slides as needed
    ];
  
    function nextSlide() {
      if (currentSlideIndex < slides.length - 1) {
        currentSlideIndex++;
      }
    }
  
    function prevSlide() {
      if (currentSlideIndex > 0) {
        currentSlideIndex--;
      }
    }
  </script>
  
  <style>
    .slideshow {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 400px;
      background-color: #222;
      color: #fff;
      font-family: Arial, sans-serif;
      border-radius: 10px;
      overflow: hidden;
      position: relative;
    }
  
    .slide {
      display: flex;
      width: 100%;
      height: 100%;
    }
  
    .slide img {
      width: 50%;
      object-fit: contain;
      border-right: 2px solid #fff;
    }
  
    .slide-content {
      padding: 20px;
      width: 50%;
      display: flex;
      justify-content: center;
      align-items: center;
      text-align: center;
    }
  
    .arrow {
      position: absolute;
      top: 50%;
      transform: translateY(-50%);
      font-size: 2rem;
      background-color: rgba(0, 0, 0, 0.5);
      color: #fff;
      border: none;
      cursor: pointer;
      padding: 10px;
      border-radius: 50%;
    }
  
    .arrow.left {
      left: 10px;
    }
  
    .arrow.right {
      right: 10px;
    }
  </style>
  
  <div class="slideshow">
    <button class="arrow left" on:click={prevSlide}>&lt;</button>
    <div class="slide">
      <img src={slides[currentSlideIndex].image} alt="Slide Image" />
      <div class="slide-content">
        <p>{slides[currentSlideIndex].text}</p>
      </div>
    </div>
    <button class="arrow right" on:click={nextSlide}>&gt;</button>
  </div>
  