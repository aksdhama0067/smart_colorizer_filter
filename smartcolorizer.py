import argparse
import os
import cv2
import numpy as np


def init_colorizer():
    # Dynamic absolute path routing to prevent CWD errors
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    proto_path = os.path.join(base_dir, "colorization_deploy_v2.prototxt")
    model_path = os.path.join(base_dir, "colorization_release_v2.caffemodel")
    hull_path = os.path.join(base_dir, "pts_in_hull.npy")

    if not (os.path.exists(proto_path) and os.path.exists(model_path) and os.path.exists(hull_path)):
        print("[Error] Missing required deep learning model files.")
        return None, None

    # Load the Caffe model(v2)
    
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    pts = np.load(hull_path, allow_pickle=True)

    # Setup cluster centers
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313, 1, 1], 2.606, dtype="float32")]

    return net, pts


def process_image(img_path, net, pts, args):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Warning] Could not load image: {img_path}")
        return

    # Keep a copy of the original for comparison (converted to 3-channel BGR)
    gray_version = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # Only run neural network colorization if the flag is passed
    if args.colorize and net is not None:
        scaled = img.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2Lab)

        # Resize to network dimensions
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
        L_orig = cv2.split(cv2.cvtColor(scaled, cv2.COLOR_BGR2Lab))[0]

        colorized = cv2.merge([L_orig, ab])
        colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
        colorized = np.clip(colorized * 255, 0, 255).astype("uint8")
    else:
        colorized = img

    # --- THE NEW SIDE-BY-SIDE STITCHING LOGIC ---
    if args.compare:
        # np.hstack joins the black & white image and the colorized image side-by-side
        final_output = np.hstack([gray_version, colorized])
    else:
        final_output = colorized

    # Output naming setup
    if args.output and not os.path.isdir(args.input):
        out_path = args.output
    else:
        base, ext = os.path.splitext(os.path.basename(img_path))
        suffix = "_compared" if args.compare else "_colorized"
        out_path = f"{base}{suffix}{ext}"

    cv2.imwrite(out_path, final_output)
    print(f"[Success] Processed and saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Smart Filter & Colorizer")
    parser.add_argument("-i", "--input", required=True, help="Path to input image or directory folder")
    parser.add_argument("--colorize", action="store_true", help="Enable deep learning colorization")
    parser.add_argument("--compare", action="store_true", help="Output a side-by-side before/after image")
    parser.add_argument("-o", "--output", help="Output file name (ignored for batch folders)")
    args = parser.parse_args()

    net, pts = init_colorizer()

    # --- THE NEW BATCH PROCESSING LOOP ---
    if os.path.isdir(args.input):
        print(f"Scanning folder: {args.input}")
        valid_exts = (".jpg", ".jpeg", ".png")
        for filename in os.listdir(args.input):
            if filename.lower().endswith(valid_exts):
                full_path = os.path.join(args.input, filename)
                process_image(full_path, net, pts, args)
    else:
        process_image(args.input, net, pts, args)


if __name__ == "__main__":
    main()
