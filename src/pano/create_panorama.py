import cv2
import os

def create_panorama(image_paths, output_file):
    """
    Stitches multiple images into a panorama.
    
    Args:
        image_paths (list): List of paths to the input images.
        output_file (str): Path to save the output panorama image.

    Returns:
        np.ndarray: The stitched panorama image, or None if stitching fails.
    """
    # Load all valid images from the paths provided
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Cannot read {path}. Skipping this image.")

    # Check if there are enough images for stitching
    if len(images) < 2:
        print("Error: Not enough images to create a panorama. At least 2 are required.")
        return None

    # Create a Stitcher object
    stitcher = cv2.Stitcher_create()

    # Stitch images together
    status, panorama = stitcher.stitch(images)

    # Handle the result of stitching
    if status == cv2.Stitcher_OK:
        print("Stitching successful. Saving panorama...")
        cv2.imwrite(output_file, panorama)
        print(f"Panorama saved as {output_file}")
        return panorama
    else:
        print(f"Error: Stitching failed with status code {status}.")
        return None


if __name__ == "__main__":
    # Example usage
    input_images_folder = "./input_images/"  # Adjust path if needed
    output_panorama_file = "./output_panorama.jpg"

    # Gather all image files from the input folder
    image_paths = [
        os.path.join(input_images_folder, fname)
        for fname in os.listdir(input_images_folder)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    # Create the panorama
    create_panorama(image_paths, output_panorama_file)
