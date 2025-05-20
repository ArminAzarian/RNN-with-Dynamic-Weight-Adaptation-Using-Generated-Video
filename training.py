import torch
import torch.optim as optim
from model_definitions import VAE, Discriminator, combined_loss, WeightPredictorRNN, create_synthetic_video
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Hyperparameters (adjust as needed) ---
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
VIDEO_HEIGHT = 128
VIDEO_WIDTH = 128
CHANNELS = 3
NUM_FRAMES = 16

# RNN Hyperparameters
RNN_INPUT_SIZE = 3 * VIDEO_HEIGHT * VIDEO_WIDTH  # Flattened image
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
WEIGHT_UPDATE_SIZE = 1344  # Replace with your calculated value (number of parameters in VAE to adjust)

# --- Data Loading ---
transform = transforms.Compose([
    transforms.Resize((VIDEO_HEIGHT, VIDEO_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset = ImageFolder(root='data', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
def train_rnn_weight_predictor_generated_video(device="cpu"):
    # --- Model Initialization ---
    vae = VAE().to(device)
    discriminator = Discriminator().to(device)
    weight_predictor = WeightPredictorRNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, RNN_NUM_LAYERS, WEIGHT_UPDATE_SIZE).to(device)

    # --- Optimizers ---
    optimizer_vae = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_weight_predictor = optim.Adam(weight_predictor.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(dataloader, leave=True)
        for idx, (real_images, _) in enumerate(loop):
            real_images = real_images.to(device)

            # --- Train Discriminator ---
            optimizer_discriminator.zero_grad()
            generated_images, mu, log_var = vae(real_images)
            loss_discriminator = combined_loss(real_images, generated_images, discriminator, device)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # --- Train Generator (VAE and Weight Predictor) ---
            optimizer_vae.zero_grad()
            optimizer_weight_predictor.zero_grad()

            # 1. Generate images with *current* VAE weights
            generated_images, mu, log_var = vae(real_images)

            # 2. Create synthetic video from generated image
            synthetic_video = create_synthetic_video(generated_images) #Shape (B, T, C, H, W)

            # 3. Reshape video for RNN input
            B, T, C, H, W = synthetic_video.shape
            video_sequence = synthetic_video.view(B, T, -1)  # (B, T, C*H*W)
            # 4. Get weight updates from RNN
            weight_updates = weight_predictor(video_sequence)

            # 5. Apply weight updates to VAE
            with torch.no_grad():  # Disable gradient calculation during weight update
                start_index = 0
                #Modify self.enc1 weight
                layer = vae.enc1
                end_index = start_index + layer.weight.numel()
                layer.weight.data += weight_updates[:, start_index:end_index].reshape(layer.weight.shape)
                start_index = end_index
                #Modify self.enc1 bias
                layer = vae.enc1
                end_index = start_index + layer.bias.numel()
                layer.bias.data += weight_updates[:, start_index:end_index].reshape(layer.bias.shape)
                start_index = end_index

            # 6. Generate images with *updated* VAE weights
            generated_images, mu, log_var = vae(real_images)

            # 7. Calculate generator loss
            loss_generator = -combined_loss(real_images, generated_images, discriminator, device)
            loss_generator.backward()

            # 8. Update VAE and weight predictor
            optimizer_vae.step()
            optimizer_weight_predictor.step()

            loop.set_postfix(loss_discriminator=loss_discriminator.item(), loss_generator=loss_generator.item())

    print(f"Training complete for RNN Weight Predictor model (Generated Video Input)")

# --- Main execution ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rnn_weight_predictor_generated_video(device)
