<script lang="ts">
  import { onMount } from 'svelte';

  let currentSlideIndex = 0;

  const slides = [
    {
      image: "President_Barack_Obama.jpg", // Update with the correct path
      text: "Images are viewed by computers as arrays in the following format: (height, width, RGB values). Most images we take are extremely complex, such as this photo of President Obama, as not only is this image in HD, but it's also in color. If our images are too complex, it slows down training time tremendously, so it's our goal to reduce the complexity of images before we feed them into our Convolutional Neural Network (CNN). We can do this by making the images go through a series of transformations."
    },
    {
      image: "grayscale_image.jpg", // Update with the correct path
      text: "We can tremendously reduce the complexity of the image by converting it from RGB to grayscale. This simple transformation reduces the complexity by a factor of 3! In grayscale images, each pixel is represented by a single intensity value rather than three color values, thus simplifying the data while retaining essential information about the structure and texture of the image."
    },
    {
      image: "gaussian_blur.jpg", // Update with the correct path
      text: "The next transformation is called a Gaussian blur. This technique smooths the image by reducing noise and detail. Mathematically, it involves applying a Gaussian function to the image. The Gaussian function is defined as: \n\nG(x, y) = (1 / (2πσ²)) * exp(- (x² + y²) / (2σ²))\n\nwhere x and y are the coordinates of the pixel, and σ (sigma) is the standard deviation of the Gaussian distribution. The effect of this function is to average the pixel values in a neighborhood, with closer pixels given more weight, resulting in a blurred image. This helps in reducing high-frequency noise and is particularly useful in pre-processing steps for further analysis."
    },
    {
      image: "canny_edges.jpg", // Update with the correct path
      text: "If we know that edges are the most important feature, we can transform our images to reflect that using the Canny edge detection algorithm. This algorithm detects the boundaries within an image by looking for areas where there is a rapid change in intensity. The process involves several steps: applying a Gaussian filter to smooth the image, finding the intensity gradient of the image, applying non-maximum suppression to remove spurious response to edge detection, and using double thresholding to detect strong and weak edges, followed by edge tracking by hysteresis. This results in an image where the edges are highlighted, making it easier for our CNN to focus on the most relevant features."
    },
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
    height: 500px;
    background-color: #333; /* Dark gray background */
    color: #fff;
    font-family: 'Helvetica Neue', Arial, sans-serif;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    padding: 0 60px; /* Add padding to the container to avoid text overlap */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
  }

  .slide {
    display: flex;
    width: 100%;
    height: 100%;
  }

  .slide img {
    width: 50%;
    object-fit: contain;
  }

  .slide-content {
    padding: 20px;
    width: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: left;
    font-size: 16px;
    line-height: 1.6;
  }

  .arrow {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    font-size: 2rem;
    background-color: rgba(0, 0, 0, 0.3);
    color: #fff;
    border: none;
    cursor: pointer;
    padding: 10px;
    border-radius: 50%;
    z-index: 1; /* Ensure arrows are above other elements */
    transition: background-color 0.3s ease;
  }

  .arrow.left {
    left: 15px;
  }

  .arrow.right {
    right: 15px;
  }

  .arrow:hover {
    background-color: rgba(0, 0, 0, 0.6);
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