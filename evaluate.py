import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from attack import generate_adversial_batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print("Downloading CIFAR-10 pre-trained ResNet20...")
    model: nn.Module = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model = model.to(device)
    model.eval()
    
    
    
    print("ResNet20 model successfully loaded and set to evaluation mode")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                             std=[0.2023, 0.1994, 0.2010])
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    batch_size = 32
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0  
    )

    print(f"Dataset loaded! Total test images: {len(test_dataset)}")
    print(f"Total batches to process: {len(test_loader)}")

    total_images = 0
    clean_correct = 0
    adv_correct = 0
    saved_batch_data = None

    attack_config = {
        "epsilon": 0.03,  # Total perturbation limit
        "alpha": 0.01,    # Step size for PGD
        "num_iter": 10    # Iterations for PGD
    }
    attack_type = "fgsm" 

    print(f"Starting evaluation loop using {attack_type.upper()}...")

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        batch_size_current = images.size(0)
        total_images += batch_size_current
        
        with torch.no_grad():
            clean_outputs = model(images)
            clean_probs = F.softmax(clean_outputs, dim=1)
            clean_conf, clean_preds = torch.max(clean_probs, 1)
            clean_correct += (clean_preds == labels).sum().item()

        adv_images = generate_adversial_batch(
            model=model, 
            images=images.clone(), 
            labeles=labels, 
            attack_method=attack_type, 
            eps=attack_config["epsilon"],
            alpha=attack_config["alpha"],
            iters=attack_config["num_iter"]
        )

        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_probs = F.softmax(adv_outputs, dim=1)
            adv_conf, adv_preds = torch.max(adv_probs, 1)
            adv_correct += (adv_preds == labels).sum().item()
            
        if batch_idx == 0:
            saved_batch_data = {
                "clean_images": images.cpu().clone(),
                "adv_images": adv_images.cpu().clone(),
                "labels": labels.cpu().clone(),
                "clean_preds": clean_preds.cpu().clone(),
                "clean_conf": clean_conf.cpu().clone(),
                "adv_preds": adv_preds.cpu().clone(),
                "adv_conf": adv_conf.cpu().clone()
            }
            
        if batch_idx % 10 == 0:
            current_clean_acc = (clean_correct / total_images) * 100
            current_adv_acc = (adv_correct / total_images) * 100
            print(f"Batch {batch_idx}/{len(test_loader)} | "
                  f"Running Clean Acc: {current_clean_acc:.2f}% | "
                  f"Running Adv Acc: {current_adv_acc:.2f}%")
                  
    final_clean_acc = (clean_correct / total_images) * 100
    final_adv_acc = (adv_correct / total_images) * 100

    print("\n" + "="*40)
    print("             EVALUATION REPORT             ")
    print("="*40)
    print(f"Total Images Evaluated: {total_images}")
    print(f"Baseline Model Accuracy  : {final_clean_acc:.2f}%")
    print(f"Adversarial Accuracy     : {final_adv_acc:.2f}%")
    print(f"Attack Success Rate      : {(final_clean_acc - final_adv_acc):.2f}% degradation")
    print("="*40)

    save_path = "adversarial_batch.pt"
    torch.save(saved_batch_data, save_path)
    print(f"\nSaved sample batch to '{save_path}' for visualise.py to plot.")

if __name__ == '__main__':
    main()