import os
from generator import Generator
from discriminator import Discriminator
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils import Image_Dataset, view_image_normalised
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(image_size: int = 64, batch_size: int = 16, max_training_steps: int = 500, epochs: int = 10, grayscale: bool = False):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')
    input_size = image_size
    generator = Generator(input_size, image_size, 1, grayscale).to(device)
    discriminator = Discriminator(input_size, grayscale).to(device)

    if os.path.exists("saves/generator.pth") and os.path.exists("saves/discriminator.pth"):
        generator.load_state_dict(torch.load("saves/generator.pth"))
        discriminator.load_state_dict(torch.load("saves/discriminator.pth"))

    train_loader = DataLoader(
        Image_Dataset(
            root_dir="../datasets/animalFaces/",
            image_res=image_size,
            grayscale=grayscale
        ),
        batch_size=batch_size,
        shuffle=True
    )

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    loss_fn = torch.nn.BCELoss()

    # pre_train_steps = int(max_training_steps * 0.01)

    # for step in range(pre_train_steps):
    #     discriminator_optimizer.zero_grad()
    #     true_data = next(iter(train_loader)).to(device)
    #     true_data = true_data.view(batch_size, 1 if grayscale else 3, image_size, image_size)
    #     true_discriminator_output = discriminator(true_data)
    #     true_discriminator_loss = loss_fn(true_discriminator_output, torch.ones_like(true_discriminator_output))
    #     true_discriminator_loss.backward()
    #     print(f'Discriminator Pre-Train {step} / {pre_train_steps} - Loss: {true_discriminator_loss.item()}')
    #     discriminator_optimizer.step()
    
    for epoch in range(epochs):
        for (step, true_data) in enumerate(train_loader):
            if step >= max_training_steps or true_data.shape[0] != batch_size:
                break

            true_data = true_data.to(device)
            true_data = true_data.view(batch_size, 1 if grayscale else 3, image_size, image_size)

            generator_optimizer.zero_grad()
            noise = torch.randn(batch_size, 4096).float().to(device)
            generated_data = generator(noise)

            test_image = generated_data[0].permute(1, 2, 0).detach().cpu().numpy()

            if step % 50 == 0:
                view_image_normalised(true_data[0].permute(1, 2, 0).detach().cpu().numpy())

            if step % 10 == 0:
                view_image_normalised(test_image)

            true_labels = torch.ones(batch_size, 1).to(device)
            generated_labels = torch.zeros(batch_size, 1).to(device)

            generator_discriminator_output = discriminator(generated_data)
            generator_loss = loss_fn(generator_discriminator_output, true_labels)
            generator_loss.backward()
            generator_optimizer.step()

            discriminator_optimizer.zero_grad()
            true_discriminator_output = discriminator(true_data)
            true_discriminator_loss = loss_fn(true_discriminator_output, true_labels)

            generator_discriminator_output = discriminator(generated_data.detach())

            generator_discriminator_loss = loss_fn(generator_discriminator_output, torch.zeros_like(generator_discriminator_output))
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            print(f"Step {step} - Generator Loss: {generator_loss.item()}, Discriminator Loss: {true_discriminator_loss.item()}")

        noise = torch.randn(batch_size, 4096).float().to(device)
        final_generated_data = generator(noise)
        # plt.ioff()
        for i in range(batch_size):
            plt.imshow(final_generated_data[i].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
            plt.show()

        torch.save(generator.state_dict(), f"saves/generator.pth")
        torch.save(discriminator.state_dict(), f"saves/discriminator.pth")

if __name__ == "__main__":
    plt.ion()
    train(image_size=64, batch_size=256, max_training_steps=500, epochs=10, grayscale=True)

    # import torch
    # print(torch.backends.mps.is_available())