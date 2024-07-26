import cv2
import matplotlib.pyplot as plt

def display_images(images, titles=None):
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()

if "__main__" == __name__:
    
    image_paths = ['./ORAND-CAR-2014/CAR-A/a_train_images/a_car_000155.png',
                   "./ORAND-CAR-2014/CAR-A/a_train_images/a_car_000156.png",
                   "./ORAND-CAR-2014/CAR-A/a_train_images/a_car_000157.png"]
    images = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error loading image at path: {path}")
        else:
            images.append(img)

    if not images:
        raise ValueError("No images were successfully loaded. Please check the file paths.")

    binary_images = [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] for img in images]

    display_images(images, titles=["Original Image 1", "Original Image 2", "Original Image 3"])
    display_images(binary_images, titles=["Binary Image 1", "Binary Image 2", "Binary Image 3"])