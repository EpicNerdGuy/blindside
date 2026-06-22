# Blindside

CNN adversarial attack generator. Implements FGSM and PGD from scratch in PyTorch, no attack libraries (no CleverHans, no torchattacks), targeting a pretrained ResNet18.

## What it does
<img width="1491" height="586" alt="image" src="https://github.com/user-attachments/assets/0b651fa1-bd92-46c0-ade7-a24cab51d6ae" />


Takes a correctly classified image, computes the gradient of the loss with respect to the input, and perturbs the image just enough to flip the model's prediction while keeping the perturbation visually imperceptible.

Example: a panda classified at 57.7% confidence, after a perturbation of epsilon = 0.007 in the direction of the gradient sign, gets misclassified as a gibbon at 99.3% confidence. Same image to a human. Garbage to the model.

## Attacks implemented

**FGSM (Fast Gradient Sign Method)**
Single step. Perturbs the image in the direction of the sign of the gradient, scaled by epsilon.

```
x_adv = x + epsilon * sign(∇x J(θ, x, y))
```

**PGD (Projected Gradient Descent)**
Iterative version of FGSM. Takes multiple small steps, projecting back into the epsilon-ball after each step to keep the perturbation bounded.

## Project structure

```
blindside/
├── model.py        # ResNet18 wrapper, frozen weights, label mapping
├── attacks.py       # fgsm_attack(), pgd_attack()
├── dataset.py       # sample image loader
├── utils.py         # preprocessing, denormalization, comparison grids
├── run_fgsm.py       # CLI entry point for FGSM
├── run_pgd.py       # CLI entry point for PGD
├── evaluate.py       # batch eval, success rate, confidence shift
├── results/         # output comparison images
└── requirements.txt
```

## Usage

```bash
python run_fgsm.py --image panda.jpg --epsilon 0.007
python run_pgd.py --image panda.jpg --epsilon 0.03 --alpha 0.005 --steps 10
```

## Requirements

```
torch
torchvision
Pillow
```

## References

- Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM)
- Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks"
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)

## Status

Work in progress. Victim model: ResNet18 (pretrained, frozen).
