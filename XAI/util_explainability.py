"""
Utility module for Model Explainability
This module provides classes and functions to generate Grad-CAM heatmaps for model explainability.

Autor: Przemek Sekula (Antigravity)
Created: 2026-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.
    Reference: https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model (nn.Module): The PyTorch model to explain.
            target_layer (nn.Module): The layer to compute the activations and gradients for.
                                      Usually the last convolutional layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Ensure all model parameters require gradients for backprop to work
        for param in self.model.parameters():
            param.requires_grad = True
            
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x, class_idx=None, normalize=True):
        """
        Computes the Grad-CAM heatmap for the given input image and class.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (1, C, H, W).
            class_idx (int, optional): The class index to explain. If None, uses the predicted class.
            normalize (bool): Whether to normalize the heatmap to [0, 1]. Default True.
            
        Returns:
            np.ndarray: The generated heatmap.
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Backward pass for the specific class
        score = logits[0, class_idx]
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        # Determine sizes
        b, k, u, v = gradients.size()
        
        # Global Average Pooling of gradients (alpha values)
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        # Weighted combination of activations
        cam = (weights * activations).sum(1, keepdim=True)
        
        # Apply ReLU (we are only interested in features that have a positive influence)
        cam = F.relu(cam)
        
        # Resize to input image size
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Convert to numpy
        cam = cam.squeeze().detach().cpu().numpy()
        
        if normalize:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx

    def __call__(self, x, class_idx=None, normalize=True):
        return self.forward(x, class_idx, normalize)


def overlay_heatmap(image_tensor, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the heatmap on the original image.
    
    Args:
        image_tensor (torch.Tensor): Original image tensor of shape (1, C, H, W) or (C, H, W).
                                     Should be unnormalized for visualization if possible, 
                                     but this function handles standard normalized tensors.
        heatmap (np.ndarray): Grad-CAM heatmap (2D array).
        alpha (float): Transparency factor for the heatmap overlay.
        colormap (int): OpenCV colormap to use.
        
    Returns:
        np.ndarray: The combined image (H, W, 3) in RGB format.
    """
    
    # Process image tensor to numpy image
    if len(image_tensor.shape) == 4:
        img = image_tensor.squeeze(0)
    else:
        img = image_tensor
        
    # Un-normalize if needed (assuming standard ImageNet normalization was used)
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.uint8(255 * img)
    
    # Process heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Combine
    height, width = img.shape[:2]
    heatmap = cv2.resize(heatmap, (width, height))
    
    overlayed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    
    return overlayed_img, img


def visualize_cam_with_colorbar(heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET, vmin=None, vmax=None):
    """
    Visualizes the heatmap with a consistent scale and a colorbar.
    Returns a matplotlib figure/image instead of just an array, or returns the array 
    and handles plotting separately? 
    To make it easier, let's return the overlay image and the heatmap image normalized 
    by vmin/vmax.
    """
    # Create a figure to plot with colorbar
    # But usually we return numpy arrays for flexibility.
    # Let's handle the normalization manually here.
    
    if vmin is None: 
        vmin = heatmap.min()
    if vmax is None:
        vmax = heatmap.max()
        
    # Clip and Normalize
    norm_heatmap = np.clip(heatmap, vmin, vmax)
    norm_heatmap = (norm_heatmap - vmin) / (vmax - vmin + 1e-8)
    
    # Process original image (for overlay)
    if len(original_image.shape) == 4: # batch dim
        img = original_image.squeeze(0)
    else:
        img = original_image
        
    # Un-normalize if needed (assuming standard ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img_uint8 = np.uint8(255 * img) # RGB
    
    # Colorize heatmap
    heatmap_uint8 = np.uint8(255 * norm_heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match image
    height, width = img_uint8.shape[:2]
    heatmap_colored = cv2.resize(heatmap_colored, (width, height))
    
    # Overlay
    overlayed_img = cv2.addWeighted(heatmap_colored, alpha, img_uint8, 1 - alpha, 0)
    
    return overlayed_img, heatmap_colored, norm_heatmap


class IntegratedGradients:
    """
    Integrated Gradients implementation.
    Reference: https://arxiv.org/abs/1703.01365
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate scaled inputs from baseline (black image) to input_image
        # Shape: (steps + 1, C, H, W)
        step_list = np.arange(steps + 1) / steps
        return input_image * step_list[:, np.newaxis, np.newaxis, np.newaxis]

    def get_gradients(self, img_input, class_idx):
        img_input.requires_grad = True
        self.model.zero_grad()
        
        logits = self.model(img_input)
        
        # If class_idx is None, take the predicted class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)[0].item()
            
        score = logits[:, class_idx].sum()
        score.backward()
        
        return img_input.grad

    def generate(self, x, class_idx=None, steps=50):
        """
        Computes Integrated Gradients for the given input image.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (1, C, H, W).
            class_idx (int, optional): The class index to explain.
            steps (int): Number of steps for approximation.
            
        Returns:
            np.ndarray: The integrated gradients (attributions).
            int: The explained class index.
        """
        # Ensure model is in eval mode and zero grad
        self.model.eval()
        self.model.zero_grad()
        
        # If class_idx is None, run forward pass to get prediction
        if class_idx is None:
            with torch.no_grad():
                logits = self.model(x)
                class_idx = torch.argmax(logits, dim=1).item()
        
        # 1. Generate interpolated images
        # Move to CPU for numpy operation then back to device
        x_cpu = x.detach().cpu().numpy()
        interpolated_images = self.generate_images_on_linear_path(x_cpu, steps)
        interpolated_images = torch.tensor(interpolated_images, dtype=torch.float32, device=x.device).squeeze(1)
        
        # 2. Compute gradients for linearly interpolated images
        # We can process in batches if memory is an issue, but for now let's try all at once
        # or split if 'steps' is large.
        
        # To handle potential memory issues, let's process in small batches
        gradients_list = []
        batch_size = 10 # Adjust based on memory
        
        for i in range(0, steps + 1, batch_size):
            batch = interpolated_images[i:i + batch_size]
            grads = self.get_gradients(batch, class_idx)
            gradients_list.append(grads)
            
        total_gradients = torch.cat(gradients_list, dim=0)
        
        # 3. Approximate the integral using trapezoidal rule
        # (avg of adjacent gradients) * (step size)
        # Actually standard definition is average of gradients * (x - x_baseline)
        
        avg_gradients = torch.mean(total_gradients[:-1] + total_gradients[1:], dim=0) / 2.0
        # Wait, simple Riemann sum (avg of all gradients) is often used too.
        # Let's stick to the definition: (x - x') * Average(gradients)
        
        avg_gradients = torch.mean(total_gradients, dim=0)
        
        integrated_gradients = (x - 0) * avg_gradients # assuming baseline is 0
        
        # Convert to numpy
        attributions = integrated_gradients.squeeze().detach().cpu().numpy()
        
        return attributions, class_idx
    
    def __call__(self, x, class_idx=None, steps=50):
        return self.generate(x, class_idx, steps)


def visualize_ig(attributions, original_image):
    """
    Visualizes the Integrated Gradients attributions.
    
    Args:
        attributions (np.ndarray): Attribution map (C, H, W).
        original_image (np.ndarray): Original image (H, W, 3).
        
    Returns:
        np.ndarray: Visualization image.
    """
    # Sum across channels to get a 2D map
    # Take absolute value because negative contribution is also important
    vis = np.sum(np.abs(attributions), axis=0)
    
    # Robust Normalization: clip outliers to 99th percentile
    # This prevents a single bright pixel from making everything else dark
    v_max = np.percentile(vis, 99)
    if v_max > 0:
        vis = np.clip(vis, 0, v_max)
        vis = vis / v_max
    else:
        vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    
    # Apply colormap
    vis = np.uint8(255 * vis)
    heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    return heatmap


class OcclusionSensitivity:
    """
    Occlusion Sensitivity implementation.
    Slides a window over the image and measures the drop in confidence.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, x, class_idx=None, patch_size=40, stride=20, normalize=True):
        """
        Computes Occlusion Sensitivity map.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (1, C, H, W).
            class_idx (int, optional): The class index to explain.
            patch_size (int): Size of the square patch to occlude.
            stride (int): Stride for sliding the patch.
            normalize (bool): Whether to normalize the heatmap to [0, 1]. Default True.
            
        Returns:
            np.ndarray: The sensitivity map (H, W).
        """
        self.model.eval()
        
        # 1. Get baseline probability
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            
            if class_idx is None:
                class_idx = torch.argmax(probs, dim=1).item()
                
            baseline_prob = probs[0, class_idx].item()
        
        b, c, h, w = x.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)
        
        # 2. Slide window
        # Create a batch of occluded images to speed up inference if possible
        # For simplicity and memory safety, we'll do one by one or small batches
        
        # Generate coordinates
        h_steps = range(0, h - patch_size + 1, stride)
        w_steps = range(0, w - patch_size + 1, stride)
        
        for i in h_steps:
            for j in w_steps:
                # Create occluded image
                x_occluded = x.clone()
                x_occluded[:, :, i:i+patch_size, j:j+patch_size] = 0 # Replace with black (or mean)
                
                # Predict
                with torch.no_grad():
                    out = self.model(x_occluded)
                    prob = torch.softmax(out, dim=1)[0, class_idx].item()
                
                # Drop in probability = Importance
                # If baseline > prob, then the occluded part was important (positive diff)
                # If baseline < prob, then the occluded part was misleading (negative diff)
                diff = baseline_prob - prob
                
                # Add to heatmap (accumulate for overlapping regions)
                heatmap[i:i+patch_size, j:j+patch_size] += diff
                counts[i:i+patch_size, j:j+patch_size] += 1
                
        # Average key regions
        heatmap = heatmap / (counts + 1e-8)
        
        if normalize:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, class_idx

    def __call__(self, x, class_idx=None, patch_size=40, stride=20, normalize=True):
        return self.generate(x, class_idx, patch_size, stride, normalize)


class CounterfactualGenerator:
    """
    Counterfactual Explanation Generator.
    Uses gradient descent to find the minimal change to the input image
    that flips the model's prediction.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, x, target_class=None, steps=100, learning_rate=0.1, lambda_l2=0.01):
        """
        Generates a counterfactual image.
        
        Args:
            x (torch.Tensor): Input image tensor (1, C, H, W).
            target_class (int, optional): The class to flip to. 
                                          If None, it flips to the other class (binary).
            steps (int): Optimization steps.
            learning_rate (float): Step size.
            lambda_l2 (float): Regularization weight to keep optimization close to original.
            
        Returns:
            torch.Tensor: Counterfactual image.
            bool: Whether the flip was successful.
            int: The target class index.
        """
        # Ensure model is frozen
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get original prediction
        with torch.no_grad():
            orig_logits = self.model(x)
            orig_class = torch.argmax(orig_logits, dim=1).item()
            
        # Determine target class
        if target_class is None:
            # Assuming binary or finding ANY other class
            # For binary: 1 - orig_class
            # For multi-class: could be the second highest, or specific
            # Here we assume it's binary as per cardio_scars dataset (Good vs Inflamed)
            target_class = 1 - orig_class
            
        # Create a trainable copy of the input
        x_cf = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=learning_rate)
        
        scaler = torch.nn.CrossEntropyLoss()
        
        successful = False
        
        for i in range(steps):
            optimizer.zero_grad()
            logits = self.model(x_cf)
            
            # Loss 1: Classification Loss (maximize target class prob -> minimize CE)
            # We want to force it to be target_class
            target_tensor = torch.tensor([target_class], device=x.device)
            loss_cls = scaler(logits, target_tensor)
            
            # Loss 2: Distance Loss (L2) - keep it close to original
            loss_dist = torch.norm(x_cf - x, p=2)
            
            # Total loss
            loss = loss_cls + lambda_l2 * loss_dist
            
            loss.backward()
            optimizer.step()
            
            # Check if flipped
            pred_class = torch.argmax(logits, dim=1).item()
            if pred_class == target_class:
                successful = True
                # We could stop early, but maybe we want to minimize distance further?
                # Let's stop if confidence is reasonably high, e.g. > 0.5
                probs = torch.softmax(logits, dim=1)
                if probs[0, target_class] > 0.6:
                    break
                    
        return x_cf.detach(), successful, target_class

    def __call__(self, x, target_class=None, steps=100, learning_rate=0.1):
        return self.generate(x, target_class, steps, learning_rate)


