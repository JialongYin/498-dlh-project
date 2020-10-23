import warnings
warnings.filterwarnings("ignore")
import os
import copy
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML

from data import Dataset, collate_wrapper
from model import Generator, Discriminator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training (default=400k)')
    parser.add_argument('--every_n_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size (default=512)")
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    args = parser.parse_args()
    args.dataset = "MIMIC_CXR_dataset/"
    args.checkpoint_path = "checkpoint_emixer/"
    args.checkpoint = "checkpoint"

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    if len(args.run) == 0:
        run_count = len([dir for dir in os.listdir(args.checkpoint_path) if dir[0:3] == "run"])
        args.run = 'run{}'.format(run_count)
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.run)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    return args

def run_training(args, dataset, train_loader):
    checkpoint_path = os.path.join(args.checkpoint_path, args.checkpoint)
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # Create the generator
    netG = Generator(vocab_size=dataset.vocab_size).to(device)
    # Create the Discriminator
    netD = Discriminator(vocab_size=dataset.vocab_size).to(device)
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", ngpu, "GPUs!")
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    nz = 120
    lr_D = 2e-4
    lr_G = 5e-5
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    fixed_clss = torch.zeros((64, 14), device=device)
    # fixed_clss[:, [1, 3, 9]] = 1
    fixed_clss[:, [8]] = 1
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G)


    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    discIter = 1 # 2
    genIter = 5
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(args.epochs):
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):
            # Format batch
            real_rpts, real_imgs, real_clss = data[0].to(device), data[1].to(device), data[2].to(device)
            b_size = real_imgs.size(0)
            # Generate batch of latent vectors
            # noise = torch.randn(b_size, nz, 1, 1, device=device)
            noise = torch.zeros(b_size, nz, 1, 1, device=device)
            for j in range(b_size):
                noise[j][j][0][0] = 1
            for k in range(discIter):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # print("(1) Update D network")
                ## Train with all-real batch
                netD.zero_grad()
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output_imgs, output_rpts, output_joint = netD(real_imgs, real_rpts, real_clss)
                # Calculate loss on all-real batch
                # errD_rpts = criterion(output_rpts, label)
                errD_imgs = criterion(output_imgs, label)
                # errD_joint = criterion(output_joint, label)
                errD_real = errD_imgs# + errD_rpts + errD_joint
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output_imgs.mean().item()# + output_rpts.mean().item() + output_joint.mean().item()

                ## Train with all-fake batch
                # Generate fake image batch with G
                fake_imgs, fake_rpts = netG(noise, real_clss)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output_imgs, output_rpts, output_joint = netD(fake_imgs.detach(), fake_rpts, real_clss) #.detach()
                # Calculate D's loss on the all-fake batch
                # errD_rpts = criterion(output_rpts, label)
                errD_imgs = criterion(output_imgs, label)
                # errD_joint = criterion(output_joint, label)
                errD_fake = errD_imgs# + errD_rpts + errD_joint
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output_imgs.mean().item()# + output_rpts.mean().item() + output_joint.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

            for k in range(genIter):
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                # print("(2) Update G network")
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                fake_imgs, fake_rpts = netG(noise, real_clss)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output_imgs, output_rpts, output_joint = netD(fake_imgs, fake_rpts, real_clss)
                # Calculate G's loss based on this output
                # errG_rpts = criterion(output_rpts, label)
                errG_imgs = criterion(output_imgs, label)
                # errG_joint = criterion(output_joint, label)
                errG = errG_imgs# + errG_rpts + errG_joint
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output_imgs.mean().item()# + output_rpts.mean().item() + output_joint.mean().item()
                # Update G
                optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake_imgs, fake_rpts = netG(fixed_noise, fixed_clss)
                    fake = fake_imgs.detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
        # save model
        if epoch % 10 == 0 or epoch == args.epochs-1:
            torch.save({
            'epoch': epoch+1,
            'G model_state_dict': netG.state_dict(),
            'D model_state_dict': netD.state_dict(),
            'G optimizer_state_dict': optimizerG.state_dict(),
            'D optimizer_state_dict': optimizerD.state_dict(),
            'vocab_size': dataset.vocab_size,
            'args': vars(args)
            }, checkpoint_path)


    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())


    # Grab a batch of real images from the dataloader
    # real_batch = real_imgs
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_imgs.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

def main(args):
    global tic
    tic = time.time()
    train_dataset = Dataset(args.dataset+"train.pkl")
    print("train_dataset len:", len(train_dataset))
    print("Epoch:{} batch_size:{} learning_rate:{}".format(args.epochs, args.batch_size, args.learning_rate))
    print("run:", args.run)
    print("device:", args.device)
    print("gpu count:", torch.cuda.device_count())
    print("cpu count:", os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4*torch.cuda.device_count(), collate_fn=collate_wrapper, pin_memory=True)
    run_training(args, train_dataset, train_loader)
    print('[{:.2f}] Finish training'.format(time.time() - tic))

if __name__ == '__main__':
    args = get_args()
    main(args)
