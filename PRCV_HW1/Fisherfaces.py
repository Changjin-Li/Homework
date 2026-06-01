from utils import *


def main():
    if len(sys.argv) < 2:
        print(f"Format: {sys.argv[0]} <csv file> [out file]")
        sys.exit(1)

    output_folder = "."
    if len(sys.argv) >= 3:
        output_folder = sys.argv[2]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    fn_csv = sys.argv[1]

    try:
        images, labels = read_csv(fn_csv)
    except Exception as e:
        print(f"Error: cannot open {fn_csv}, {e}")
        sys.exit(1)

    if len(images) <= 1:
        print("Error: at least 2 images are required")
        sys.exit(1)

    height, width = images[0].shape[0], images[0].shape[1]

    test_sample = images[-1]
    test_label = labels[-1]
    train_images = images[:-1]
    train_labels = labels[:-1]

    model = cv2.face.FisherFaceRecognizer.create(num_components=16)
    model.train(train_images, np.array(train_labels))

    predicted_label, confidence = model.predict(test_sample)
    print(f"Predicted label = {predicted_label} / Actual label = {test_label}")

    eigenvalues = model.getEigenValues()
    eigenvectors = model.getEigenVectors()
    mean = model.getMean()

    mean_img = mean.reshape((height, -1))
    mean_img_norm = norm_0_255(mean_img)
    if len(sys.argv) == 2:
        cv2.imshow("mean", mean_img_norm)
    else:
        cv2.imwrite(os.path.join(output_folder, "mean.png"), mean_img_norm)

    canvas_h, canvas_w = height * 4, width * 4
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    num_fisherface = min(eigenvectors.shape[1], 16)
    for i in range(num_fisherface):
        eigenvector = eigenvectors[:, i].reshape((height, -1))
        eigen_norm = norm_0_255(eigenvector)
        eigen_color = cv2.applyColorMap(eigen_norm, cv2.COLORMAP_BONE)
        if len(sys.argv) == 2:
            cv2.imshow(f"fisherface_{i}", eigen_color)
        else:
            eigen_color_norm = norm_0_255(eigen_color)
            cv2.imwrite(os.path.join(output_folder, f"fisherface_{i}.png"), eigen_color_norm)
            canvas[(i // 4) * height: (i // 4 + 1) * height, (i % 4) * width: (i % 4 + 1) * width, :] = eigen_color_norm
        print(f"Eigenvalue #{i} = {eigenvalues[0][i]:.5f}")
    cv2.imwrite(os.path.join(output_folder, "fisherfaces.png"), canvas)

    # Image Reconstruction
    first_img = train_images[0]
    first_img_flat = first_img.flatten().reshape(1, -1).astype(np.float64)

    results = []
    for fisherface_idx in range(num_fisherface):
        evs = eigenvectors[:, fisherface_idx]
        projection = subspaceProject(evs, mean, first_img_flat)
        reconstruction = subspaceReconstruct(evs, mean, projection)
        recon_img = reconstruction.reshape((height, -1))
        recon_norm = norm_0_255(recon_img)
        if len(sys.argv) == 2:
            cv2.imshow(f"fisherface_reconstruction_{fisherface_idx}", recon_norm)
        else:
            results.append(recon_norm)
            cv2.imwrite(os.path.join(output_folder, f"fisherface_reconstruction_{num_fisherface}.png"), recon_norm)

    canvas_h, canvas_w = height * 4, width * 4
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for i in range(16):
        canvas[(i // 4) * height: (i // 4 + 1) * height, (i % 4) * width: (i % 4 + 1) * width] = results[i]
    cv2.imwrite(os.path.join(output_folder, "fisherface_reconstructions.png"), canvas)

    if len(sys.argv) == 2:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()