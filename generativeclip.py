#!git clone https://github.com/openai/CLIP.git
#https://github.com/openai/CLIP 
#CLIP (Contrastive Language Image Pre-training)
#Learning Transferable Visual Models for Natural Language Supervision
#Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
#Sandhini Agarwal, Girish Sastry, Amand Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever

#!git clone https://github.com/CompVis/taming-transformers
#https://github.com/CompVis/taming-transformers 
#Taming Transformers for High Resolution Image Synthesis
#Patrick Esser, Robin Rombach, Bjorn Ommer


## install some extra libraries 
#!pip install --no-deps ftfy regex tqdm
#!pip install omegaconf==2.0.0 pytorch-lightning==1.0.8
#!pip uninstall torchtext --yes
#!pip install einops

import numpy as np
np.Inf=np.inf
import torch, os, imageio, pdb, math
from torch import inf
#from torch import string_classes
#string_classes = str
import torchvision 
import torchvision.transforms as T 
import torchvision.transforms.functional as TF 
import collections.abc as container_abcs 
import PIL
import matplotlib.pyplot as plt 
string_classes = (str,)
int_classes = (int,)

import yaml 
from omegaconf import OmegaConf 

from CLIP import clip

#import warnings
#Warning.filterwarnings('ignore')

## helper functions 

def show_from_tensor(tensor):
   img = tensor.clone()
   img = img.mul(255).byte()
   img = img.cpu().numpy().transpose((1,2,0))
   
   plt.figure(figsize=(10,7))
   plt.axis('off')
   plt.imshow(img)
   plt.show()
  
def norm_data(data):
	return (data.clip(-1,1)+1)/2

### Parameters 
learning_rate= .5
batch_size= 1
wd = .1
noise_factor = .1 

total_iter = 300
im_shape = [255, 400, 3] #height, width, channel
size1, size2, channels = im_shape

### CLIP MODEL ###
clipmodel, _ = clip.load('ViT-B/32',jit=False)
clipmodel.eval()
print(clip.available_models())

print("clip model visual input resolution: ", clipmodel.visual.input_resolution)

device = torch.device("cuda:0")
torch.cuda.empty_cache()
## Taming transformrt instantiation

##%cd taming-transformers/

##!mkdir -p models/vqgan_imagenet_f16_16384/checkpoints
##!mkdir -p models/vqgan_imagenet_f16_16384/configs

##if len(os.listdir('models/vqgan_imagenet_f16_16384/checkpoints'))==0:
##    wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt'
##    wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'models/vqgan_imagenet_f16_16384/configs/model.yml'


from taming.models.vqgan import VQModel

def load_config(config_path, display=False):
    config_data = OmegaConf.load(config_path)
    if display:
         print(yaml.dump(OmegaConf.to_container(config_data)))
    return config_data 

def load_vqgan(config,chk_path=None):
    model = VQModel(**config.model.params)
    if chk_path is not None:
        state_dict = torch.load(chk_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model.eval()

def generator(x):
    x = taming_model.post_quant_conv(x)
    x = taming_model.decoder(x)
    return x 

taming_config = load_config("./models/vqgan_imagenet_f16_16384/configs/model.yaml", display=True)
taming_model = load_vqgan(taming_config, chk_path="./models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(device)

## Declare the values that we going to 

class Parameters(torch.nn.Module):
   def __init__(self):
       super(Parameters, self).__init__()
       self.data = torch.randn(batch_size, 256, size//16, size2//16).cuda()
       self.data = torch.nn.Parameter(torch.sin(self.data))
    
   def forward(self):
       return self.data
       
def init_params():
    params=Parameters().cuda()
    optimizer = torch.optim.AdamW([{'params':[params.data], 'lr':learning_rate}], weight_decay=wd)
    return params, optimizer
    
    
    ###Encoding prompts and a few more things
    normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258,0.27577711))
    
    def encodeText(text):
        t=clip.tokenize(text).cuda()
        t=clipmodel.encode_text(t).detach().clone()
        return t 
    
    def createEncodings(include, exclude, extras):
        include_enc=[]
        for text in include:
            include_enc.append(encodeText(text))
        exclude_enc = encodeText(exclude) if exclude !='' else 0
        extras_enc= encodeTet(extras) if extras !='' else 0
        
        return include_enc, exclude_enc, extras_enc
    augTransform = torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomAffine(30, (.2, .2), fill=0)).cuda()
    
    Params, optimizer = init_params()
    
    with torch.no_grad():
        print(Params().shape)
        img= norm_data(generator(Params().cpu()))
        print("img dimensions: ", img.shape)
        show_from_tensor(img[0])

### create crops
def create_crops(img, num_crops=30):
    p=size1/2
    img = torch.nn.functional.pad(img, (p,p,p,p), mode='constant', value=0)
    img = augTransforms(img)
    crop_set=[]
    for ch in range(num_crops):
        gap1= int(torch.normal(1.0, .5, ()).clip(.2, 1.5) * size1)
        gap2= int(torch.normal(1.0, .5, ()).clip(.2, 1.5) * size1)
        offsetx = torch.randint(0, int(size1*2-gap1),())
        offsety = torch.randint(0, int(size1*2-gap1),())
      
        crop = img[:,:,offset:offset+gap2, offsety:offsety+gap2]
      
        crop = torch.nn.functional.interpolate(crop(224,224), mode='bilinear', align_corners=True)
        crop_set.append(crop)
    
    img_crops=torch.cat(crop_set,0)  
    img_crops = img_crops + noise_factor*torch.randn_like(img_crops, requires_grad=False)
    return img_crops
    
    
def showme(Params, show_crop):
    with torch.no_grad():
        generated = generator(Params())
        
        if (show_crop):
            print("Augmented cropped example")
            aug_gen = generated.float();
            aug_gen = create_crops(aug_gen, num_crops=1)
            aug_gen_norm = norm_data(aug_gen[0])
            show_from_tensor(aug_gen_norm)
        
        print("Generation")
        latest_gen=norm_data(generated.cpu())
        show_from_tensor(latest_geo[0])
    return (latest_gen[0])
    
def optimize_result(Params, prompt):
    alpha=1
    beta=.5
    
    out=generator(Params())
    out=norm_data(out)
    out=create_crops(out)
    out=normalize(out)
    image_enc=clipmodel.encode_image(out)
    
    final_enc = w1*prompt + w1*extras_enc 
    final_text_include_enc = final_enc/final_enc.norm(dim=-1,keepdim=True)
    final_text_exclude_enc = exclude_enc 
    
    main_loss = torch.cosine_similarity(final_text_include_enc, image_enc, 1)# 30
    penalize_loss = torch.cosine_similarity(final_text_exclude_enc, image_enc, -1)# 30
    
    final_loss = alpha*main_loss + beta*penalize_loss
    
    return final_loss 
    
def optimize(Params, optimizer, prompt):
    loss = optimize_result(Params, promt).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    
    
 #training loop
def training_loop(Params, optimizer, show_crop=False):
    res_img=[]
    res_z=[] 
     
    for prompt in include_enc:
        iteration=0
        Params, optimizer = init_params()
         
        for it in range(total_iter):
            loss = optimize(Params, optimizer, prompt)
            if iteration>0 and iteration%(total_iter-1) == 0:
                 new_img = showme(Params, show_crop)
                 res_img.append(new_img)
                 res_z.append(Params())
                 print("loss:", loss.item(), "\iteration:",iteration)
            iteration+=1
        torch.cuda.empty_cache()
    return res_img, res_z
 
torch.cuda.empty_cache()
include=['sketch of a lady','sketch of a man on a horse']
exclude='watermark, cropped, confusing, incoherent, cut, blurry'
extras= "watercolor paper texture"
w1=1
w2=2
include_enc, exclude_enc, extras_enc = createEncodings(include, exclude, extras)
res_img, res_z = training_loop(Params, optimizer, show_crop=True)