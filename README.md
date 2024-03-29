# Landscape-Convolutional-GAN

A deep convolutional generative adversarial network that I trained to generate images of landscapes with 3 color channels.
This is based on the concept of an adversarial network, but with the Generator and Discriminator being convolutional networks, which are better suited for dealing with images.
<p align="center">
  <img src="/DCGAN.jpg?raw=true" width="800" height="238"/>
</p>

[Image Source](https://www.microsoft.com/en-us/research/blog/how-can-generative-adversarial-networks-learn-real-life-distributions-easily/)

The loss of the Discriminator is based on whether the input image to it was in the training set, or came from the Generator. The loss of the Generator is based on the Discriminator's output/label, in a sense, the more its able to fool the Discriminator, the less it gets its weight updated via gradient descent because its doing a "good job", but if it gets discovered by the Discriminator, it gets a larger weight update because it doing a bad "job". This process continues in the training loop untill the discriminator can't tell the difference between real and generated images (it picks the label of each with $p=0.5$).

$L_D = \mathop{max}_{D} ; \space log(D(x)) + log(1-D(G(z)))$

$L_G = \mathop{min}_{G} ; \space log(1-D(G(z)))$

Where $x$ is an image from the trianing set, $z$ is random noise from the latent space that the Generator uses to produce images, $D$ is the Discriminator, $G$ is the Generator. This is just the **_binary cross entropy loss_**.

The loading of the training dataset, definition og both generator and discriminator networks, the training loop, as well a s the function used for loading the trained weights and generating images are all in the **_dcgan.py_** file.

The code is commented to describe what each step does. Some of the commments (depending on their length) are to the right of the code blocks.

### The trinaing losses:

<p align="center">
  <img src="/training_losses.png?raw=true" width="800" height="444"/>
</p>


### Example of what the network would output:

<p align="center">
  <img src="/DCGAN_example_output.jpg?raw=true" width="512" height="201"/>
</p>
