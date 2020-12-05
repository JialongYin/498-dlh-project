import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF
import torch

imgs_ls = os.listdir("./real_imgs")
print(imgs_ls)
real_imgs_ls = []
for img_file in imgs_ls:

    img = Image.open("./real_imgs/"+img_file)
    img = TF.resize(img, (128, 128))
    img = TF.to_tensor(img)
    real_imgs_ls.append(img)
real_imgs = torch.stack(real_imgs_ls)

imgs_ls = os.listdir("./x_rays_pgan_output")
print(imgs_ls)
fake_imgs_ls = []
for img_file in imgs_ls:
    img = Image.open("./x_rays_pgan_output/"+img_file)
    img = TF.resize(img, (128, 128))
    img = TF.to_tensor(img)
    fake_imgs_ls.append(img)
fake_imgs = torch.stack(fake_imgs_ls)


# Grab a batch of real images from the dataloader
# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_imgs[64:128], padding=5, normalize=True).cpu(),(1,2,0)))
# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake_imgs[64:128], padding=2, normalize=True),(1,2,0)))
# plt.show()
plt.savefig('./real_fake_img.png')
