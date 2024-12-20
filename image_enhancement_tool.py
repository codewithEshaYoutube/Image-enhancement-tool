from PIL import Image, ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt

def adjust_brightness(image_path, factor):
    """ Adjust the brightness of an image """
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(factor)  # factor > 1 brightens, factor < 1 darkens
    return img_bright

def adjust_contrast(image_path, factor):
    """ Adjust the contrast of an image """
    img = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(factor)  # factor > 1 increases contrast
    return img_contrast

def adjust_sharpness(image_path, factor):
    """ Adjust the sharpness of an image """
    img = Image.open(image_path)
    enhancer = ImageEnhance.Sharpness(img)
    img_sharp = enhancer.enhance(factor)  # factor > 1 sharpens, factor < 1 softens
    return img_sharp

def adjust_saturation(image_path, factor):
    """ Adjust the saturation of an image """
    img = Image.open(image_path)
    enhancer = ImageEnhance.Color(img)
    img_saturation = enhancer.enhance(factor)  # factor > 1 increases saturation
    return img_saturation

def adjust_ai_contrast(image_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    """ Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast """
    # Load image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Convert image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge back and convert to BGR
    limg = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return img_clahe

def display_images(original, enhanced, title1="Original", title2="Enhanced"):
    """ Display the original and enhanced images side by side """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    axes[1].imshow(enhanced)
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    plt.show()

def main():
    image_path = 'input_image.jpg'  # Replace with your image path

    # Apply basic brightness and contrast adjustments
    img_bright = adjust_brightness(image_path, 1.5)  # Example: Brightness up
    img_contrast = adjust_contrast(image_path, 2.0)  # Example: Contrast up
    
    # Apply AI-based contrast enhancement
    img_ai_contrast = adjust_ai_contrast(image_path, clip_limit=2.0)
    
    # Display original vs adjusted images
    original_image = Image.open(image_path)
    display_images(np.array(original_image), np.array(img_bright), title1="Original", title2="Brightness Enhanced")
    display_images(np.array(original_image), np.array(img_contrast), title1="Original", title2="Contrast Enhanced")
    display_images(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), img_ai_contrast, title1="Original", title2="AI Contrast Enhanced")

if __name__ == "__main__":
    main()

