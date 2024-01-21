import numpy as np
from scipy.special import erf, expit
from skimage import io, exposure
import matplotlib.pyplot as plt

def illumination_boost(X, lambda_val):
    # Convert image to double, even if it is already a double
    X = exposure.rescale_intensity(X)
    
    I1 = (np.max(X) / np.log(np.max(X) + 1)) * np.log(X + 1)
    I2 = 1 - np.exp(-X)
    I3 = (I1 + I2) / (lambda_val + (I1 * I2))
    I4 = erf(lambda_val * np.arctan(np.exp(I3)) - 0.5 * I3)
    I5 = (I4 - np.min(I4)) / (np.max(I4) - np.min(I4))

    return I5

# Load image
I = io.imread('夜间\\room.jpg').astype(float) / 255.0
plt.imshow(I)
lambda_val = 2
out_image = illumination_boost(I, lambda_val)

# Display the original and processed images side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(I)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(out_image)
plt.title('Processed Image')

plt.show()