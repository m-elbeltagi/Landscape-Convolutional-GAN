import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import datetime
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


## this implementation follows some (not all because a few didn't lead to convergence when training) tips & tricks from the 'ganhack' github post, experimentally tested techniques that improve performance 

## for saving the trained model params
save_path = r'C:\Users\' ## put your save path here

train_image_path = r'' ## put the training forlder path here

## setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Device in use: {} \n'.format(device))


## to look at images, 3 color channels, values between 0,255
def load_image(path) :
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype="int32" )
    return data


torch.manual_seed(42) 
np.random.seed(42)
image_size = 256            ## training images have different dimension, so resize them to be the same
colour_channels = 3
z = 100                     ## latent space vector input to generator
gen_channels = 64           ## size of feature maps in generator
dis_channels = 64          ## size of feature maps in discriminator
           
learning_rate = 0.0002
n_epochs = 15
batch_size = 50             ##(number of training samples divisible by batch_size? got weird error when it wasnt)

beta1 = 0.5             ## ADAM optimizer paramter, used for computing running avgs of gradients


## normalize transform to normalize the 0-255 pixels to be between [-1,1], when loaded range is automatically [0,1], so the shift & scale value is just themidway point, so 0.5
train_dataset = ImageFolder(root=train_image_path, transform=transforms.Compose([transforms.Resize(size=image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# plot some training images after transformations(change seed to see diferent images)
def plot_transformed_images(train_loader):
    real_batch = next(iter(train_loader))
    plt.figure(figsize=(50, 50))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))




## to initalize weights distributed according to Gaussian, .find() searchs for its string argument, and if it finds nothing returns -1
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


## next 2 functions to calculate dimensions of images after convs & transpose convs
def calc_transposeConv_out(I, K, S, P):
    print ( (I-1)*S - 2*P + K)
    
## this also calculates avg pooling output    
def calc_conv_out(I, K, S, P):
    print ( ((I - K + 2*P)/S) + 1)




## defining generator
class GeneratorNetwork(nn.Module):
    def __init__(self):
        super(GeneratorNetwork, self).__init__() 
        
        ## (1, 100, 1, 1)--->(1, gen_channels*64, 8, 8)
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(in_channels=z, out_channels=gen_channels*8, kernel_size=16, stride=1, padding=0, bias=False),                ## transposed conv is perferred to classical upsampling techniques because it's learnable, however due to the transpose conv matrix having zero values could lead to "checkerboard effect" in output, so to mitigate we can either upsample using a Pixel Shuffle layer, or a classical upsampling followed by convolution 
                                    nn.BatchNorm2d(gen_channels*8),                                                                                                ## this layer standardizes its input based on the mean/stdev of current batch, then applies linear transformation, when evaluationg uses avg of all the batch means/stdevs, reduces covariate shift, and might have a regularization effect as well, takes as input number of output channels of previous layer, outputs tensor with same shape as input
                                    nn.Mish(inplace=True))                                                                                                          ## Mish is state of the art fpr computer vision (for other tasks try swish first, better tested) ## CANCELLED (PReLU failed in training): trying to learn the negative part of the leaky relu (to avoid sparse gradients, same reason as not using maxpool in this network) instead of fixing it to a value like 0.01, this activation func should not be used with weight decay for good performance, is this true for other regularization techniques equivalent to weight decay (or for example smooth labels, which have regularization effect)? not sure but try it out
                                    
        
        ## (1, gen_channels*8, 16, 16)--->(1, gen_channels*8, 32, 32)                   
        self.layer2 =  nn.Sequential(nn.ConvTranspose2d(in_channels=gen_channels*8, out_channels=gen_channels*8, kernel_size=4, stride=2, padding=1, bias=False),               
                                     nn.BatchNorm2d(gen_channels*8),
                                     nn.Mish(inplace=True)) 
        
        
        ## (1, gen_channels*8, 32, 32)--->(1, gen_channels*4, 64, 64) 
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(in_channels=gen_channels*8, out_channels=gen_channels*4, kernel_size=4, stride=2, padding=1, bias=False), 
                                    nn.BatchNorm2d(gen_channels*4),
                                    nn.Mish(inplace=True))
        
        
        ## (1, gen_channels*4, 64, 64)--->(1, gen_channels*4, 128, 128)
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(in_channels=gen_channels*4, out_channels=gen_channels*4, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(gen_channels*4),
                                    nn.Mish(inplace=True))
        

        ## (1, gen_channels*4, 128, 128)--->(1, colour_channels, 256, 256)
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(in_channels=gen_channels*4, out_channels=colour_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.Tanh())                                                                                                                    ## recommended to use tanh as the last layer of the generator
                             
        
        
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x
        
          



## defining discriminator
class DiscriminatorNetwork(nn.Module):
    def __init__(self):
        super(DiscriminatorNetwork, self).__init__() 
        
        ## (batch_size, colour_channels, 256, 256)--->(batch_size, dis_channels, 128, 128)                                                                  ## removing batch norm from this first layer seems to help with training the discriminator, also tried using PReLU, but training was failing
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=colour_channels, out_channels=dis_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        
        ## (batch_size, dis_channels, 128, 128)--->(batch_size, dis_channels*2, 64, 64)
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=dis_channels, out_channels=dis_channels*2, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(dis_channels*2),
                                    nn.LeakyReLU(0.2, inplace=True))     


        ## (batch_size, dis_channels*2, 64, 64)---->(batch_size, dis_channels*4, 32, 32)                     
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=dis_channels*2, out_channels=dis_channels*4, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(dis_channels*4),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        
        ## (batch_size, dis_channels*4, 32, 32)--->(batch_size, dis_channels*8, 16, 16) 
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=dis_channels*4, out_channels=dis_channels*8, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(dis_channels*8),
                                    nn.LeakyReLU(0.2, inplace=True))


        ## (batch_size, dis_channels*8, 16, 16)--->(batch_size, dis_channels*16, 4, 4)
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=dis_channels*8, out_channels=dis_channels*16, kernel_size=10, stride=2, padding=0, bias=False),
                                    nn.BatchNorm2d(dis_channels*16),
                                    nn.LeakyReLU(0.2, inplace=True))
        
        ## takes flattened tensor (batch_size, dis_channels*16, 4, 4)--1st_linear-->(batch_size, 1, 1, 1)
        self.layer6 = nn.Sequential(nn.Conv2d(in_channels=dis_channels*16, out_channels=1, kernel_size=4, stride=4, padding=0, bias=False),
                                    nn.Sigmoid())    
        
        
        
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x
   

           
generator = GeneratorNetwork()
generator.to(device)
generator.apply(weights_init)


discriminator = DiscriminatorNetwork()
discriminator.to(device)
discriminator.apply(weights_init)

optimizerG = torch.optim.Adam(params=generator.parameters(), lr=learning_rate, betas=(beta1, 0.999)) 
optimizerD = torch.optim.Adam(params=discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999)) 
loss_func = nn.BCELoss()
                  



def Dtrain_step(real_batch, fake_batch, labels_tensor):
    ## using noisy (and smooth) labels, flipping with probability of 5%, noisy helps with regularization (and helps with keeping gradients non-zero in the beginning), smooth helps with robustness against adversarial attacks
    if np.random.uniform(low=0, high=1) > 0.05:
        real_label = np.random.uniform(low=0.8, high=1)
        fake_label = np.random.uniform(low=0.0, high=0.2)
    else: 
        real_label = np.random.uniform(low=0.0, high=0.2)
        fake_label = np.random.uniform(low=0.8, high=1)


    
         
    discriminator.train()
    
    labels_tensor.fill_(real_label)
    
    ## forward pass real batch through discriminator, calculate gradients (separating into real and fake batches works well in practice)
    discriminator.zero_grad()
    y_hat = discriminator(real_batch).view(-1)
    Dloss_real = loss_func(y_hat, labels_tensor)
    Dloss_real.backward()
    D_x = y_hat.mean().item()
    
    ## forward pass fake batch through discriminator, accumulate gradients
    labels_tensor.fill_(fake_label)
    y_hat = discriminator(fake_batch.detach()).view(-1)
    Dloss_fake = loss_func(y_hat, labels_tensor)
    Dloss_fake.backward()
    D_G_z1 = y_hat.mean().item()
    optimizerD.step()
    
    
    return Dloss_real.item() + Dloss_fake.item(), D_x, D_G_z1


def Gtrain_step(fake_batch, labels_tensor):
    real_label = 1
    labels_tensor.fill_(real_label)
    
    generator.train()
    
    generator.zero_grad()
    y_hat = discriminator(fake_batch).view(-1)
    Gloss = loss_func(y_hat, labels_tensor)
    Gloss.backward()
    D_G_z2 = y_hat.mean().item()
    optimizerG.step()
    
    
    return Gloss.item(), D_G_z2


   
G_losses = []
D_losses = []    

def start_train_loop(save_model=True):
    print('start time is: {} \n'.format(datetime.datetime.now()))
    for epoch in range(n_epochs):
        
        for i, data in enumerate(train_loader, start=0):

            real_images = data[0].to(device)
            
            latent_noise = torch.randn(size=(batch_size, z, 1, 1), device=device)
            fake_images = generator(latent_noise)
            labels_tensor = torch.zeros(size=(batch_size,), dtype=torch.float, device=device)
            
            
            D_loss, D_x, D_G_z1 = Dtrain_step(real_images, fake_images, labels_tensor)
            G_loss, D_G_z2 = Gtrain_step(fake_images, labels_tensor)
            
            G_losses.append(G_loss)
            D_losses.append(D_loss)
            
            if i % 50 == 0:
                print ('[epoch: {}/{}] [iteration: {}/{}] D_loss: {}, G_loss: {}, D(x): {}, D(G(z)): {}/{}'.format(epoch+1, n_epochs, i, len(train_loader), D_loss, G_loss, D_x, D_G_z1, D_G_z2))
            
            
    if save_model == True:
        torch.save(generator.state_dict(), save_path + r'\generator_trained_weights.pt')
        torch.save(discriminator.state_dict(), save_path + r'\discriminator_trained_weights.pt')
    print('Finsih time is: {} \n'.format(datetime.datetime.now()))
            
        
def plot_losses():
    plt.figure(dpi=1000)  
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()            
    
    
def generate_images(n_images):
    load_generator = GeneratorNetwork()
    load_generator.load_state_dict(torch.load(save_path + r'\generator_v4_trained_weights.pt'))
    load_generator.to(device)
    load_generator.eval() 
    
    
    torch.manual_seed(1215) 

    noise = torch.randn(size=(n_images, z, 1, 1), device=device)
    images = load_generator(noise)

    plt.figure(dpi=600)
    plt.figure(figsize=(100, 100))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(vutils.make_grid(images[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))




# start_train_loop(save_model=True)   
# plot_losses()    

generate_images(1)
