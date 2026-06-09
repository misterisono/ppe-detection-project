import cv2
import matplotlib.pyplot as plt


def show_annotated_image(results, title="Inference Result"):
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_frame)
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()