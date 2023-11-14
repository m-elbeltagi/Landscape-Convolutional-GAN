# DCGAN
A deep convolutional generative adversarial network that I trained to generate images of faces with 3 color channels (and with some modifications to the architecture, later trained to generate images of landscapes).
This is based on the concept of an adversarial network, but with the generator and discriminator networks being composed of convolutional layers, which are better suited for generating images.
<p align="center">
  <img src="/DCGAN.png?raw=true" width="400" height="300"/>
</p>
The loading of the training dataset, definition og both generator and discriminator networks, the training loop, as well a s the function used for loading the trained weights and generating images are all in the **dcgan.py** file.
The code is commented to describe what each step does. Some of the commments (depending on their length) are to the right of the code blocks.
