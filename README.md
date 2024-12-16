# MRI Generation Algorithm
A missing MRI sequence generation algorithm based on CycleGAN (Generative Adversarial Network). The algorithm embeds attention mechanisms of different dimensions on top of residual blocks to extract sequence features. The generator and discriminator learn the feature mapping relationships between sequences to generate missing sequences.

# Train:

`cyclegan` (main file): Includes the model, training process, and saving of generated images.

`dataloader_XXX`: Reads data. Output format: return `{"A": item_a, "B": item_b, "label": int(label)}`

`models`: Defines the generator and discriminator models in CycleGAN.

`pic_cut`: Crops the DWI and DCE modality images according to the tumor location in the mask.


# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* Python == 3.10.14
* numpy == 1.26.4