import os
from generator import Generator_conv, Generator_linear, Generator_conv_simple
from discriminator import Discriminator_conv, Discriminator_linear
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import Image_Dataset, plot_confidences

def train(image_size: int = 256, batch_size: int = 16, max_training_steps: int = 500, epochs: int = 10, grayscale: bool = True):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    # generator = Generator_linear(input_size=image_size * image_size).to(device)
    # discriminator = Discriminator_linear(input_size=image_size * image_size).to(device)
    generator = Generator_conv_simple(channels=1 if grayscale else 3).to(device)
    discriminator = Discriminator_conv(channels=1 if grayscale else 3, image_size=image_size).to(device)

    fig1 = plt.figure()
    fig2 = plt.figure()

    discriminator_learning = True
    generator_learning = True
    
    if os.path.exists("saves/generator.pth") and os.path.exists("saves/discriminator.pth"):
        if input("load? (y/n)") == "y":
            generator.load_state_dict(torch.load("saves/generator.pth"))
            discriminator.load_state_dict(torch.load("saves/discriminator.pth"))
            losses = torch.load("saves/losses.pth")
            generator_losses = losses["generator_losses"]
            discriminator_losses = losses["discriminator_losses"]
            reference_discriminator_losses = losses["reference_discriminator_losses"]
            generator_discriminator_losses = losses["generator_discriminator_losses"]
        else:
            generator_losses = []
            discriminator_losses = []
            reference_discriminator_losses = []
            generator_discriminator_losses = []

    train_data = Image_Dataset(root_dir="../datasets/animalFaces", image_res=image_size, grayscale=grayscale)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):            
        for (step, reference_data) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if step >= max_training_steps or reference_data.shape[0] != batch_size:
                break

            reference_data = reference_data.to(device)
            reference_data = reference_data.view(batch_size, 1 if grayscale else 3, image_size, image_size)

            generator_learning = True

            if len(generator_losses) > 0:
                if generator_losses[-1] > 1:
                    discriminator_learning = True
                else:
                    discriminator_learning = True

            if generator_learning:
                generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, 1 if grayscale else 3, image_size, image_size).float().to(device)
            noise = (noise - torch.min(noise)) / (torch.max(noise) - torch.min(noise))
            generated_data = generator(noise)

            reference_labels = torch.ones(batch_size, 1).to(device)
            generated_labels = torch.zeros(batch_size, 1).to(device)

            generator_discriminator_output = discriminator(generated_data.detach())
            generator_loss = loss_fn(generator_discriminator_output, reference_labels)

            if generator_learning:
                generator_loss.backward(retain_graph=True)
                generator_optimizer.step()


            if discriminator_learning:
                discriminator_optimizer.zero_grad()
            reference_discriminator_output = discriminator(reference_data)
            reference_discriminator_loss = loss_fn(reference_discriminator_output, reference_labels)

            generator_discriminator_loss = loss_fn(generator_discriminator_output, generated_labels)
            discriminator_loss = (reference_discriminator_loss + generator_discriminator_loss) / 2

            # Compute gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, reference_data, generated_data, device)
            # Add gradient penalty to the discriminator loss
            discriminator_loss += 10 * gradient_penalty

            if discriminator_learning:
                discriminator_loss.backward()
                discriminator_optimizer.step()

            plot_confidences(fig2, generator_discriminator_output, reference_discriminator_output)

            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())
            reference_discriminator_losses.append(reference_discriminator_loss.item())
            generator_discriminator_losses.append(generator_discriminator_loss.item())

            # figure 1
            fig1.clf()
            ax1 = fig1.add_subplot(1, 1, 1)
            ax1.plot(generator_losses, label="Generator Loss", color="red")
            ax1.plot(discriminator_losses, label="Discriminator Loss", color="blue")
            ax1.plot(reference_discriminator_losses, label="Reference Discriminator Loss", color="green")
            ax1.plot(generator_discriminator_losses, label="Generator Discriminator Loss", color="yellow")
            ax1.legend()
            plt.show()
            plt.pause(0.01)


        if epoch % 5 == 0:
            noise = torch.randn(batch_size, 1 if grayscale else 3, image_size, image_size).float().to(device)
            noise = (noise - torch.min(noise)) / (torch.max(noise) - torch.min(noise))
            final_generated_data = generator(noise)

            ax1.set_ylim(0, image_size)
            # for i in range(batch_size):
            for i in range(10):
                ax1.imshow(final_generated_data[i].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                plt.show()
                plt.pause(0.1)
            
        if epoch == 0:
            ax1.set_ylim(0, image_size)
            # for i in range(batch_size):
            for i in range(min(10, len(reference_data))):
                if grayscale:
                    ax1.imshow(reference_data[i].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
                else:
                    ax1.imshow(reference_data[i].detach().cpu().numpy())
                plt.show()
                plt.pause(0.1)

        torch.save(generator.state_dict(), f"saves/generator.pth")
        torch.save(discriminator.state_dict(), f"saves/discriminator.pth")
        losses = {
            "generator_losses": generator_losses,
            "discriminator_losses": discriminator_losses,
            "reference_discriminator_losses": reference_discriminator_losses,
            "generator_discriminator_losses": generator_discriminator_losses
        }
        torch.save(losses, f"saves/losses.pth")


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

if __name__ == "__main__":
    plt.ion()
    train(image_size=128, batch_size=256, max_training_steps=1000, epochs=10000, grayscale=False)