<script>
    export let show = false;
    export let onClose;
  </script>
  
  {#if show}
    <div class="popup">
      <div class="close" on:click={onClose}>✖</div>
      <h2>Input Info</h2>
      <div class="images-container">
        <img src="og_6.png" alt="Image 6" class="image-left" />
        <img src="og_0.png" alt="Image 0" class="image-right" />
      </div>

      <p>These two images are from the MNIST dataset, which contains 70,000 28x28 images of handwritten numbers 0 through 9. The image on the left is an
        example of the number 6 from the MNIST dataset and the image on the right is an example of the number 0 from the dataset.
        <br><br>
        Sometimes, 6s can look like 0s and 0s can look like 6s if written poorly, but we can use the power of computer vision to create a model that will accurately
        discern the differences between the two. As you click through the layers in the graphic, you can see how these images are broken down as they pass through the model.
        Since these images are already in black and white color scale, and are not that complex, we don't need to perform any 
        transfomrations before feeding them into the model.
        <br><br>
        In this example, we have labeled Class 1 for 6, and class 0 for 0
        </p>
    </div>
  {/if}
  
  <style>
    .popup {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background-color: black;
      color: white;
      padding: 20px;
      border: 1px solid white;
      z-index: 10;
      width: 800px;
      height: 600px;
      overflow-y: auto;
    }
  
    .close {
      position: absolute;
      top: 10px;
      right: 10px;
      cursor: pointer;
      color: red;
      border: 1px dotted red;
      padding: 2px 5px;
      font-size: 18px;
      font-weight: bold;
    }
  
    .images-container {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 20px;
    }
  
    .images-container img {
      max-width: 45%;
      height: auto;
      border: 1px solid white;
    }
  
    pre {
      background-color: #333;
      padding: 10px;
      border-radius: 5px;
      overflow-x: auto;
    }
  
    code {
      color: #00ff00;
    }
  </style>  