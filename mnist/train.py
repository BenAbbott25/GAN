import os
from generator import Generator_conv, Generator_linear, Generator_conv_simple
from discriminator import Discriminator_conv, Discriminator_linear
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loadData import load_data

def train(image_size: int = 28, batch_size: int = 16, max_training_steps: int = 500, epochs: int = 10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    # generator = Generator_linear(input_size=image_size * image_size).to(device)
    # discriminator = Discriminator_linear(input_size=image_size * image_size).to(device)
    generator = Generator_conv(channels=1).to(device)
    discriminator = Discriminator_conv(channels=1).to(device)

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
    train_data, train_labels, test_data, test_labels = load_data()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        # discriminator, discriminator_optimizer, generator_losses, discriminator_losses, reference_discriminator_losses, generator_discriminator_losses = pretrain(
        #     discriminator=discriminator,
        #     train_loader=train_loader,
        #     discriminator_optimizer=discriminator_optimizer,
        #     loss_fn=loss_fn,
        #     image_size=image_size,
        #     batch_size=batch_size,
        #     max_training_steps=max_training_steps / 10,
        #     device=device,
        #     epochs=1,
        #     generator_losses=generator_losses,
        #     discriminator_losses=discriminator_losses,
        #     reference_discriminator_losses=reference_discriminator_losses,
        #     generator_discriminator_losses=generator_discriminator_losses
        # )
            
        for (step, reference_data) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if step >= max_training_steps or reference_data.shape[0] != batch_size:
                break

            reference_data = reference_data.to(device)
            reference_data = reference_data.view(batch_size, 1, image_size, image_size)

            generator_learning = True

            if len(generator_losses) > 0:
                if generator_losses[-1] > 1:
                    discriminator_learning = True
                else:
                    discriminator_learning = True

            if generator_losses[-1] > 0.5:
                discriminator_learning = False
            else:
                discriminator_learning = True

            if generator_learning:
                generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, 1, image_size, image_size).float().to(device)
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
            # discriminator_loss = (3*generator_discriminator_loss + reference_discriminator_loss) / 4

            if discriminator_learning:
                discriminator_loss.backward()
                discriminator_optimizer.step()

            # print(f"Step {step} - Generator Loss: {generator_loss.item()}, Discriminator Loss: {discriminator_loss.item()}")
            # figure 2
            fig2.clf()
            ax2 = fig2.add_subplot(1, 1, 1)
            suptitle = f"Generator Learning: {'Enabled' if generator_learning else 'Disabled'} | Discriminator Learning: {'Enabled' if discriminator_learning else 'Disabled'}"
            fig2.suptitle(suptitle)
            x = np.linspace(0, len(generator_discriminator_output), len(generator_discriminator_output))
            generator_discriminator_output = np.concatenate(sorted(generator_discriminator_output.detach().cpu().numpy()))
            reference_discriminator_output = np.concatenate(sorted(reference_discriminator_output.detach().cpu().numpy()))
            ax2.plot(generator_discriminator_output, label="Generator Discriminator Output", color="red")
            ax2.plot(reference_discriminator_output, label="Reference Discriminator Output", color="blue")

            ax2.fill_between(x, generator_discriminator_output, -1, label="Generator Discriminator Output", color="pink")
            ax2.fill_between(x, reference_discriminator_output, 1, label="Reference Discriminator Output", color="lightblue")
            # average
            average_generator_discriminator_output = np.mean(generator_discriminator_output)
            average_reference_discriminator_output = np.mean(reference_discriminator_output)
            ax2.axhline(average_generator_discriminator_output, color="pink", linestyle="--")
            ax2.axhline(average_reference_discriminator_output, color="lightblue", linestyle="--")
            ax2.legend()
            ax2.set_ylim(0, 1)
            # plt.show()
            # plt.pause(0.01)

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
            noise = torch.randn(batch_size, 1, image_size, image_size).float().to(device)
            noise = (noise - torch.min(noise)) / (torch.max(noise) - torch.min(noise))
            final_generated_data = generator(noise)

            ax2.set_ylim(0, 28)
            # for i in range(batch_size):
            for i in range(10):
                ax2.imshow(final_generated_data[i].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
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

def pretrain(
        discriminator,
        train_loader: torch.utils.data.DataLoader,
        discriminator_optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        image_size: int = 28,
        batch_size: int = 16,
        max_training_steps: int = 500,
        device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
        epochs: int = 1,
        generator_losses: list = [],
        discriminator_losses: list = [],
        reference_discriminator_losses: list = [],
        generator_discriminator_losses: list = []
    ):
    discriminator.to(device)

    for (step, reference_data) in enumerate(train_loader):
        if step >= max_training_steps or reference_data.shape[0] != batch_size:
            break

        reference_data = reference_data.to(device)
        reference_data = reference_data.view(batch_size, 1, image_size, image_size)

        discriminator_optimizer.zero_grad()
        reference_discriminator_output = discriminator(reference_data)
        reference_discriminator_loss = loss_fn(reference_discriminator_output, torch.ones_like(reference_discriminator_output))
        reference_discriminator_loss.backward()
        discriminator_optimizer.step()

        print(f"Pretrain Step {step} / {max_training_steps} - Discriminator Loss: {reference_discriminator_loss.item()}")
        
        generator_losses.append(generator_losses[-1] if len(generator_losses) > 0 else 0)
        discriminator_losses.append(discriminator_losses[-1] if len(discriminator_losses) > 0 else 0)
        reference_discriminator_losses.append(reference_discriminator_losses[-1] if len(reference_discriminator_losses) > 0 else 0)
        generator_discriminator_losses.append(generator_discriminator_losses[-1] if len(generator_discriminator_losses) > 0 else 0)

    return discriminator, discriminator_optimizer, generator_losses, discriminator_losses, reference_discriminator_losses, generator_discriminator_losses

if __name__ == "__main__":
    plt.ion()
    train(image_size=28, batch_size=1024, max_training_steps=1000, epochs=10000)