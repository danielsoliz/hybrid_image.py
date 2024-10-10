import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and preprocess images
im1_path = 'images/woman_happy.png'       
im2_path = 'images/woman_neutral.png'     

im1 = cv2.imread(im1_path)
im2 = cv2.imread(im2_path)

if im1 is None or im2 is None:
    raise FileNotFoundError("One or both images not found.")

im1 = cv2.resize(im1, (512, 512))
im2 = cv2.resize(im2, (512, 512))

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian Blur
kernel_size = (31, 31)
sigma = 5

im1_blur = cv2.GaussianBlur(im1_gray, kernel_size, sigma)
im2_blur = cv2.GaussianBlur(im2_gray, kernel_size, sigma)

# Step 3: Obtain the detail image
im2_gray_float = im2_gray.astype(np.float32)
im2_blur_float = im2_blur.astype(np.float32)

im2_detail = im2_gray_float - im2_blur_float

im2_detail_norm = cv2.normalize(im2_detail, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
im2_detail_norm = im2_detail_norm.astype(np.uint8)

# Step 4: Create the hybrid image
im1_blur_float = im1_blur.astype(np.float32)

hybrid_float = im1_blur_float + im2_detail

hybrid_norm = cv2.normalize(hybrid_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
hybrid_norm = hybrid_norm.astype(np.uint8)

cv2.imwrite('hybrid.png', hybrid_norm)

# Display the hybrid image
plt.figure(figsize=(8, 8))
plt.imshow(hybrid_norm, cmap='gray')
plt.title('Hybrid Image')
plt.axis('off')
plt.show()

# Function to display the hybrid image at different scales
def display_hybrid_scales(image):
    plt.figure(figsize=(12, 6))
    scales = [1, 0.5, 0.25, 0.125]
    for i, scale in enumerate(scales):
        resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        plt.subplot(1, len(scales), i+1)
        plt.imshow(resized, cmap='gray')
        plt.title(f'Scale {scale}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display the hybrid image at different scales
display_hybrid_scales(hybrid_norm)
