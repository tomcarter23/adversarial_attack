# Adversarial Attack
A library for conducting Adversarial Attacks on pytorch image classifier models.

## Overview
Adversarial Attack is a Python library that provides a simple API and CLI for conducting adversarial attacks on PyTorch image classifier models. The library supports both standard and targeted attacks using the Fast Gradient Sign Method (FGSM) algorithm (https://arxiv.org/abs/1412.6572). 

Given a pre-trained PyTorch model and an input image, the library generates an adversarial image that is misclassified by the model but looks almost identical to the original image. 

The library comes with a set of pre-trained PyTorch models (e.g., ResNet18, ResNet50) and utility functions for loading images, preprocessing images. However users can also use their own models and images but must include their own preprocessing and loading steps (see *Running via API* section).

## Installation
Adversarial Attack can be installed by first cloning the repository and the installing dependecies using pip. It is reccomended to use a virtual environment to install dependencies.

```
git clone git@github.com:tomcarter23/adversarial_attack.git
cd adversarial_attack
python -m venv venv
source venv/bin/activate
pip install -e . 
```

Developers of the project can install extra dependencies that allow the running of the testing suite

```
pip install -e ".[test]" 
```

## Running via CLI

You can run the adversarial attack from the command line interface (CLI). Here's the general syntax:

```bash
python -m adversarial_attack --model <MODEL_NAME> --mode <MODE> --image <IMAGE_PATH> --category-truth <TRUE_CATEGORY> --category-target <TARGET_CATEGORY> --epsilon <EPSILON> --max-iterations <MAX_ITER> --output <OUTPUT_PATH>
```
### Parameters:

- `--model, -m`: The model to attack (e.g., `resnet18`, `resnet50`).
- `--mode`: The type of attack:
  - `standard`: Standard FGSM attack.
  - `targeted`: Targeted FGSM attack (default).
- `--image, -i`: Path to the input image to attack.
- `--category-truth, -c`: The true class label of the image (e.g., `cat`).
- `--category-target, -ct`: The target class label for the targeted attack (only required for targeted mode).
- `--epsilon, -eps`: The epsilon value for the attack (default: `1.0e-3`).
- `--max-iterations, -it`: Maximum number of iterations for the FGSM attack (default: `50`).
- `--output, -o`: Path to save the resulting adversarial image.


## Running via API
You can also use the library via the provided API to perform adversarial attacks programmatically on any PyTorch model.

Example usage:

import torch
from adversarial_attack.api import perform_attack
from adversarial_attack.resnet_utils import load_model_default_weights, get_model_categories, load_image, preprocess_image

```
# Load the model from provided in resnet_utils or use your own
model = resnet_utils.load_model_default_weights(model_name='resnet18')

# Load and preprocess the image using provided resnet_utils functions or use your own
image = resnet_utils.load_image('path/to/image.jpg')
image_tensor = resnet_utils.preprocess_image(image)

# Define the categories for the model
categories = resnet_utils.get_model_categories('resnet18')

# Perform the attack
result_image_tensor = perform_attack(
    mode='targeted',  # or 'standard'
    model=model,
    image=image_tensor,
    categories=categories,
    true_category='cat',
    target_category='dog',  # required only for targeted attack
    epsilon=1.0e-3,
    max_iter=50
)

result_image = Image.fromarray(np.uint8(255 * to_array(result_image_tensor)))

# Save the resulting adversarial image
if result_image is not None:
    result_image.save('path/to/output.jpg')
```

## Examples

The `sample_images/imagenet` directory contains a set of example images from the ILSCVR2012 Imagenet validation dataset which constitute part of the same dataset that the pre-trained models were trained on. 
The images are named according to their true class label (e.g., `lawn_mower_ILSVRC2012_val_00020327.JPEG`), where the true class label is the part of the filename before the `ILSVRC2012` identifier. 
True classes in each of the provided models do not contain underscores e.g. `lawn mower`. This format should be used if using these sample images for testing.

```bash
python -m adversarial_attack --model resnet50 --mode targeted --image sample_images/imagenet/lawn_mower_ILSVRC2012_val_00020327.JPEG --category-truth "lawn mower" --category-target goldfish --epsilon 1.0e-3 --max-iterations 50 --output output.jpg
```