import math
import numpy as np
from PIL import Image

image_path = r"C:\Users\colin\OneDrive\Documents\linear_transformation\image.jpg"

# Load the image
# Was having issues with an image with color, so I greyscaled it.
image = Image.open(image_path).convert("L")
image_np = np.array(image)
original_image = Image.fromarray(image_np)
print("Original Image & Matrix Shown First:")
# original_matrix = np.array(image)
original_matrix = np.array([[1, 0],[0,1]])
print (original_matrix)
original_image.show()

# Function to scale an image
def scale_image(image_np, scale_factors):

    height, width = image_np.shape
    scale_height, scale_width = scale_factors

    new_height = int(height * scale_height)
    new_width = int(width * scale_width)

    scaled_image = np.zeros((new_height, new_width), dtype=image_np.dtype)

    for y in range(new_height):
        for x in range(new_width):

            originalX = int(x / scale_width)
            originalY = int(y / scale_height)

            if originalX < width and originalY < height:
                scaled_image[y, x] = image_np[originalY, originalX]

    return scaled_image

# Scaling (T1)
scale_factors = (0.5, 0.5)
# scaled_matrix = (np.array([[0.5, 0],[0,0.5]]))
scaled_matrix = (np.array(0.5 * original_matrix))
print("1) Standard Matrix of T1 is:")
print(scaled_matrix)
image1_np = scale_image(image_np, scale_factors)
print(f'The dimension of the image_np is {image1_np.shape}')
image1 = Image.fromarray(image1_np)
image1.show()

# Function to reflect an image
def reflect_image(image_np, reflection_matrix):
    height, width = image_np.shape

    centerX = width / 2
    centerY = height / 2

    reflected_image = np.zeros_like(image_np)

    for y in range(height):
        for x in range(width):
            translated_x = x - centerX
            translated_y = -(y - centerY)

            reflected_coords = reflection_matrix @ np.array([translated_x, translated_y])
            reflectedX, reflectedY = reflected_coords

            reflectedX += centerX
            reflectedY = -reflectedY + centerY

            reflectedX = int(round(reflectedX))
            reflectedY = int(round(reflectedY))

            if 0 <= reflectedX < width and 0 <= reflectedY < height:
                reflected_image[reflectedY, reflectedX] = image_np[y, x]
    return reflected_image

# Reflection (T2, T3)
reflection_matrix1 = np.array([[0.3, 0.4], [0.4, -0.3]])
print("2) Standard Matrix of T2 reflected across y = 2x is:")
print(reflection_matrix1)
reflection_matrix2 = np.array([[-0.02, 0.64], [-0.64, -0.02]])
print("3) Standard Matrix of T2 reflected across y = -1/2x is:")
print(reflection_matrix2)
image2_np = reflect_image(image1_np, reflection_matrix1)
print(f'The dimension of the image2_np is {image2_np.shape}')
image2 = Image.fromarray(image2_np)
image2.show()

image3_np = reflect_image(image2_np, reflection_matrix2)
print(f'The dimension of the image3_np is {image3_np.shape}')
image3 = Image.fromarray(image3_np)
image3.show()

# Function to rotate an image
def rotate_image(image_np, rotation_matrix):
    height, width = image_np.shape
    centerX = width / 2
    centerY = height / 2

    rotated_image = np.zeros_like(image_np)

    for y in range(height):
        for x in range(width):
            translatedX = x - centerX
            translatedY = -(y - centerY)

            rotated_coords = rotation_matrix @ np.array([translatedX, translatedY])
            rotatedX, rotatedY = rotated_coords

            rotatedX += centerX
            rotatedY = -rotatedY + centerY

            rotatedX = int(round(rotatedX))
            rotatedY = int(round(rotatedY))

            if 0 <= rotatedX < width and 0 <= rotatedY < height:
                rotated_image[rotatedY, rotatedX] = image_np[y, x]
    return rotated_image

# Function to find the identity matrix
def findMatrix(matrix2, matrix1):
    return matrix2 @ matrix1

# Function to compute the inverse matrix
def inverse_transformation(matrix_T):
    return matrix_T.T


# Transformation (T = T2 ◦ T1)
matrixT = findMatrix(reflection_matrix1, scaled_matrix)
print("4) Standard Matrix of T = T2 o T1:")
print(matrixT)
image_T_np = reflect_image(image1_np, matrixT)
print(f'The dimension of the image_T_np is {image_T_np.shape}')
image_T = Image.fromarray(image_T_np)
image_T.show()

# Inverse Transformation (T^{-1})
matrixT_inverse = np.array([[2.4, 3.2],[3.2, -2.4]])
print("5) Standard Matrix of T^(-1):")
print(matrixT_inverse)
image_inverse_np = reflect_image(image3_np, matrixT_inverse)
print(f'The dimension of the image_inverse_np is {image_inverse_np.shape}')
image_inverse = Image.fromarray(image_inverse_np)
image_inverse.show()

# Rotation / Favorite Transformation
theta = math.pi / 4
rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
print("6) Standard rotation matrix (rotated 45 degrees):")
print(rotation_matrix)
image4_np = rotate_image(image_np, rotation_matrix)
print(f'The dimension of the image4_np is {image4_np.shape}')
image4 = Image.fromarray(image4_np)
image4.show()
