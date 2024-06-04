<script>
    export let show = false;
    export let onClose;
  </script>
  
  {#if show}
    <div class="popup">
      <div class="close" on:click={onClose}>✖</div>
      <p>CL Info Here</p>
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
      width: 300px; /* Adjust width as needed */
      height: 200px; /* Adjust height as needed */
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
  </style>
  