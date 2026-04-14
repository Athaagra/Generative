# import libraries 
import torch, pdb
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST 
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# visualization function 
def show(tensor, ch=1, size=(28,28), num=16):
   #tensor: 128 x 784
  data=tensor.detach().cpu().view(-1,ch,*size)
  grid = make_grid(data[:num], nrow=4).permute(1,2,0)
  plt.imshow(grid)
  plt.show()


#setup of the main parameters and hyperparametes
epochs = 300
cur_step = 0
info_step = 300
mean_disc_loss = 0
mean_gen_loss = 0


z_dim = 64
lr = 0.000001
loss_function = nn.BCEWithLogitsLoss()


bs = 128 #batch size
device = "cuda"


dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()), shuffle=True ,batch_size=bs)

#number of steps = 6000 / 128 = 468.75


# declare our models 


# Generator 
def genBlock(inp, out):
  return nn.Sequential(
   nn.Linear(inp,out),
   nn.BatchNorm1d(out),
   nn.ReLU(inplace=True)
)


class Generator(nn.Module):
   def __init__(self, z_dim=64, i_dim=784, h_dim=128):
       super().__init__()
       self.gen = nn.Sequential(
            genBlock(z_dim, h_dim),
            genBlock(h_dim, h_dim*2),
            genBlock(h_dim*2, h_dim*4),
            genBlock(h_dim*4, h_dim*8),
            nn.Linear(h_dim*8, i_dim),
            nn.Sigmoid(),
       )
   def forward(self, noise): 
          return self.gen(noise)


def gen_noise(number, z_dim):
   return torch.randn(number, z_dim).to(device)


# Discriminator
def discBlock(inp,out):
   return nn.Sequential(
     nn.Linear(inp,out),
     nn.LeakyReLU(0.2)
)


class Discriminator(nn.Module):
   def __init__(self, i_dim=784, h_dim=256):
       super().__init__()
       self.disc= nn.Sequential(
          discBlock(i_dim, h_dim*4),
          discBlock(h_dim*4, h_dim*2),
          discBlock(h_dim*2, h_dim),
          nn.Linear(h_dim, 1)
      )
   def forward(self,image):
       return self.disc(image)
       
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(),lr=lr)




##Training Process
x,y = next(iter(dataloader))
print(x.shape, y.shape)
print(y[:10])


noise = gen_noise(bs, z_dim)
fake = gen(noise)
#show(fake)




#calculating the loss 


#generator loss


def calc_gen_loss(loss_func, gen, disc, number, z_dim):
    noise = gen_noise(number, z_dim)
    fake = gen(noise)
    pred = disc(fake)
    targets = torch.ones_like(pred)
    gen_loss=loss_func(pred,targets)
    return gen_loss


def calc_disc_loss(loss_func, gen, disc, number, real, z_dim):
   noise = gen_noise(number, z_dim)
   fake = gen(noise)
   disc_fake = disc(fake.detach())
   disc_fake_targets = torch.zeros_like(disc_fake)
   disc_fake_loss = loss_func(disc_fake, disc_fake_targets)


   disc_real = disc(real)
   disc_real_targets = torch.ones_like(disc_real)
   disc_real_loss = loss_func(disc_real, disc_real_targets)


   disc_loss=(disc_fake_loss+disc_real_loss)/2


   return disc_loss




# Each step in going to process 128 image = size of the batch (except the last step)


for epoch in range(epochs):
   for real, _ in tqdm(dataloader):
     disc_opt.zero_grad()
     curs_bs = len(real)
     real = real.view(curs_bs, -1)
     real = real.to(device)
    
     disc_loss = calc_disc_loss(loss_function,gen,disc,curs_bs,real,z_dim)
     disc_loss.backward(retain_graph=True)
     disc_opt.step()
    
     ### generator
     gen_opt.zero_grad()
     gen_loss = calc_gen_loss(loss_function, gen, disc, curs_bs,z_dim)
     gen_loss.backward(retain_graph=True)
     gen_opt.step()
     
     #### visualization & stats
     mean_disc_loss+=disc_loss.item()/info_step
     mean_gen_loss+=gen_loss.item()/info_step
     
     if cur_step % info_step == 0 and cur_step > 0:
        fake_noise = gen_noise(curs_bs,z_dim)
        fake = gen(fake_noise)
        #show(fake)
        #show(real)
        print(f"{epoch}:step {cur_step}/ Gen loss: {mean_gen_loss} / disc_loss: {mean_disc_loss}")
        mean_gen_loss, mean_disc_loss=0, 0
     cur_step+=1
