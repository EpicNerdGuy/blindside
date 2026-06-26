import torch
import numpy as np
import matplotlib.pyplot as plt

CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def plot_adversarial_examples(data_path="adversarial_batch.pt", num_images=4):
    
    print(f"Loading data from {data_path}...")
    try:
        data = torch.load(data_path, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Could not find '{data_path}'. Make sure you run evaluate.py first!")
        return

    clean_images = data["clean_images"]
    adv_images = data["adv_images"]
    true_labels = data["labels"]
    
    clean_preds = data["clean_preds"]
    clean_conf = data["clean_conf"]
    
    adv_preds = data["adv_preds"]
    adv_conf = data["adv_conf"]

    successful_attack_indices = []
    for i in range(len(true_labels)):
        if clean_preds[i] == true_labels[i] and adv_preds[i] != true_labels[i]:
            successful_attack_indices.append(i)
            
    if len(successful_attack_indices) == 0:
        print("No successful attacks found in this batch to visualize.")
        return

    if len(successful_attack_indices) < num_images:
        print(f"Warning: Only found {len(successful_attack_indices)} successful attacks in this batch.")
        num_images = min(num_images, len(successful_attack_indices))

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3.5 * num_images), squeeze=False)
    fig.suptitle("BlindSide: Adversarial Attack Results", fontsize=18, fontweight='bold', y=1.02)

    for row, idx in enumerate(successful_attack_indices[:num_images]):
        
        # We no longer need denormalize()! Just permute and convert to numpy.
        clean_img_np = clean_images[idx].detach().permute(1, 2, 0).numpy()
        adv_img_np = adv_images[idx].detach().permute(1, 2, 0).numpy()
        
        noise_np = (adv_img_np - clean_img_np) * 10
        noise_np = np.clip(noise_np + 0.5, 0, 1)

        true_class = CIFAR_CLASSES[true_labels[idx]]
        clean_class = CIFAR_CLASSES[clean_preds[idx]]
        adv_class = CIFAR_CLASSES[adv_preds[idx]]
        
        clean_pct = clean_conf[idx].item() * 100
        adv_pct = adv_conf[idx].item() * 100

        ax = axes[row, 0]
        ax.imshow(clean_img_np, interpolation='nearest')
        ax.set_title(f"Clean Image", fontsize=12, fontweight='semibold')
        ax.set_xlabel(f"True: {true_class}\nPred: {clean_class}\nConf: {clean_pct:.1f}%", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        ax.imshow(noise_np, interpolation='nearest')
        ax.set_title("Perturbation Noise (x10)", fontsize=12, fontweight='semibold')
        ax.set_xlabel("Adversarial Gradient", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 2]
        ax.imshow(adv_img_np, interpolation='nearest')
        ax.set_title("Adversarial Image", fontsize=12, color='darkred', fontweight='bold')
        ax.set_xlabel(f"Pred: {adv_class}\nConf: {adv_pct:.1f}%", fontsize=11, color='darkred', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(h_pad=3.5) 
    fig.subplots_adjust(top=0.90)
      
    plt.savefig("blindside_results.png", bbox_inches='tight', dpi=150)
    print("Plot saved successfully to 'blindside_results.png'!")
    plt.show()

if __name__ == "__main__":
    plot_adversarial_examples()