import os
import sys
import argparse
import cv2
import numpy as np


def init_colorizer():
    """Loads and initializes the Caffe Deep Learning model for colorization."""
    import os

    # Get the exact folder where smart_filter.py is saved
    base_dir = os.path.dirname(os.path.abspath(__file__))

    proto_path = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
    model_path = os.path.join(base_dir, "colorization_release_v2.caffemodel")
    hull_path = os.path.join(base_dir, "pts_in_hull.npy")

    if not (os.path.exists(proto_path) and os.path.exists(model_path) and os.path.exists(hull_path)):
        print("[Error] Missing required deep learning model files for colorization.")
        print("Please ensure 'colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel',")
        print("and 'pts_in_hull.npy' are in the current working directory.")
        sys.exit(1)

    # Load network
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    pts = np.load(hull_path, allow_pickle=True)

    # Add cluster centers to the network
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts.transpose().reshape(2, 313, 1, 1).astype("float32")]
    net.getLayer(conv8).blobs = [np.full((1, 313), 2.606, dtype="float32")]

    return net


def colorize_image(image_path, net):
    """Colorizes a black and white image using the deep learning network."""
    print("[...] Colorizing image using AI...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Could not read image: {image_path}")
        return None

    # Normalize and convert to lab space
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize the Lab image to 224x224
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # Mean subtraction

    # Pass the L channel through the network to predict 'a' and 'b' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize predicted 'ab' volume back to original image dimensions
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Reconstruct the original image
    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)

    # Convert back to BGR and scale to 8-bit unsigned integer
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized


def apply_filter(image_path, filter_type):
    """Applies standard aesthetic photographic filters."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Could not read image: {image_path}")
        return None

    print(f"[...] Applying filter: '{filter_type}'...")

    if filter_type == 'sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        inv_blur = cv2.bitwise_not(blur)
        return cv2.divide(gray, inv_blur, scale=256.0)

    elif filter_type == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)

    elif filter_type == 'vintage':
        # Increase warmth, add a vignette effect
        rows, cols = img.shape[:2]
        # Create vignette mask
        X_resultant_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols, rows))
        mask_x = cv2.getGaussianKernel(cols, cols / 2)
        mask_y = cv2.getGaussianKernel(rows, rows / 2)
        mask = mask_y * mask_x.T
        mask = mask / mask.max()

        vintage = np.copy(img)
        # Shift colors to warmer tones
        vintage[:, :, 0] = np.clip(vintage[:, :, 0] * 0.9, 0, 255)  # Blue
        vintage[:, :, 2] = np.clip(vintage[:, :, 2] * 1.1, 0, 255)  # Red
        # Apply vignette mask
        for i in range(3):
            vintage[:, :, i] = vintage[:, :, i] * mask
        return vintage.astype(np.uint8)

    elif filter_type == 'hdr':
        # Simulates a high-dynamic-range detail enhancement
        return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

    return img


def main():
    parser = argparse.ArgumentParser(description="Smart Image Colorizer and Filter CLI Tool")

    # Arguments
    parser.add_argument("-i", "--input", required=True, help="Path to the input image file")
    parser.add_argument("-o", "--output", help="Path to save the output image (Defaults to 'output.jpg')")

    # Mutually exclusive group so users choose either automatic colorization OR a standard filter
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--colorize", action="store_true", help="Use AI to colorize a Black & White image")
    group.add_argument("--filter", choices=['sketch', 'sepia', 'vintage', 'hdr'],
                       help="Apply a creative styling filter")

    args = parser.parse_args()

    # Determine output name if not provided
    output_path = args.output if args.output else "output.jpg"

    # Process based on selection
    if args.colorize:
        net = init_colorizer()
        result = colorize_image(args.input, net)
    else:
        result = apply_filter(args.input, args.filter)

    # Save output
    if result is not None:
        cv2.imwrite(output_path, result)
        print(f"[🎉 Success] Saved processed image to: {output_path}")


if __name__ == "__main__":
    main()