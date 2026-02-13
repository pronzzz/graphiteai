import argparse
import os
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import sys

# Add parent directory to path to import backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.model.generator import Generator
from backend.model.discriminator import Discriminator
from training.dataset import ImageDataset
from training.losses import GANLoss

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
parser.add_argument("--pretrained_weights", type=str, default="", help="path to pretrained generator weights")
opt = parser.parse_args()

def train():
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = GANLoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if opt.epoch != 0:
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    
    # Load Pretrained Weights if specified (overrides epoch loading for Generator)
    if opt.pretrained_weights:
        if os.path.exists(opt.pretrained_weights):
            print(f"Loading pretrained weights from {opt.pretrained_weights}")
            try:
                # Try loading directly
                generator.load_state_dict(torch.load(opt.pretrained_weights))
            except Exception as e:
                print(f"Direct load failed: {e}. Attempting to assume weights are under 'state_dict' or key mismatch...")
                # Could add more robust loading logic here for HF models
        else:
            print(f"Pretrained weights file not found: {opt.pretrained_weights}")

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    transforms_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataloader = DataLoader(
        ImageDataset("training/data/%s" % opt.dataset_name, transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Model inputs
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            # We construct them after getting the output to match the shape dynamically
            
            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            
            # GAN Loss
            valid = Variable(Tensor(np.ones(pred_fake.size())), requires_grad=False)
            fake = Variable(Tensor(np.zeros(pred_fake.size())), requires_grad=False)
            
            loss_GAN = criterion_GAN(pred_fake, True) # True for Real Label but criterion_GAN handles it internally or we pass tensor?
            # My criterion_GAN implementation: 
            # def __call__(self, prediction, target_is_real):
            #     target_tensor = self.get_target_tensor(prediction, target_is_real)
            #     return self.loss(prediction, target_tensor)
            
            # So I don't need 'valid' tensor explicitly if I use criterion_GAN correctly.
            # But let's check losses.py. 
            # It uses self.real_label expanded to prediction shape.
            
            # So:
            loss_GAN = criterion_GAN(pred_fake, True)

            
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + 100 * loss_pixel # lambda = 100

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, True)

            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, False)

            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
            )

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))

if __name__ == "__main__":
    import numpy as np
    train()
