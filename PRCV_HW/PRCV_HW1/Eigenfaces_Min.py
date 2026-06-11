import matplotlib.pyplot as plt
import numpy as np

from utils import *


def similarity(X, Y):
    X = X.flatten()
    Y = Y.flatten()
    return np.sqrt(np.sum((X - Y) ** 2)) / np.sqrt(np.sum(X ** 2))

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

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(train_images, np.array(train_labels))

    predicted_label, confidence = model.predict(test_sample)
    print(f"Predicted label = {predicted_label} / Actual label = {test_label}")

    eigenvalues = model.getEigenValues()
    eigenvectors = model.getEigenVectors()
    mean = model.getMean()

    # Image Reconstruction
    first_img = train_images[0]
    first_img_flat = first_img.flatten().reshape(1, -1).astype(np.float64)

    results = []
    index_min = -1
    max_components = min(eigenvectors.shape[1], 400)
    for num_components in range(max_components):
        evs = eigenvectors[:, :num_components]
        projection = subspaceProject(first_img_flat, mean, evs)
        reconstruction = subspaceReconstruct(evs, mean, projection)
        sim = similarity(first_img_flat, reconstruction)
        results.append(similarity(first_img_flat, reconstruction))
        if sim < 0.05 and index_min == -1:
            index_min = num_components

    plt.plot(range(max_components), results)
    plt.xlabel("Number of Components")
    plt.ylabel("Diff")
    if index_min > -1:
        plt.axvline(x=index_min, color='r', linestyle='--')
        plt.axhline(y=results[index_min], color='r', linestyle='--')
    plt.savefig(os.path.join(output_folder, "eigenfaces_min.png"))
    plt.show()
    print(f"Minimum components = {index_min}")

    if len(sys.argv) == 2:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()