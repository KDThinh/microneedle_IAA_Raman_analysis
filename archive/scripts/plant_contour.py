import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_plant_edges(image_path):
    # 1. Read and Normalize
    img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read the image at: {image_path}")
    img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 2. Blur (Crucial for edge detection to avoid picking up camera noise)
    blurred = cv2.GaussianBlur(img_8u, (7, 7), 0)

    # 3. Canny Edge Detection (Replicating the ImageJ approach)
    # The numbers 30 and 100 are the lower and upper hysteresis thresholds.
    edges = cv2.Canny(blurred, 30, 100)

    # 4. Connect the edges 
    # Canny gives thin outlines. We dilate them so the leaves connect into a solid shape.
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 5. Find Contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 6. Filter out the background lines
    plant_contour = None
    max_area = 0
    height, width = img.shape
    
    for cnt in contours:
        # Fill the contour to get a solid area calculation, not just the line length
        area = cv2.contourArea(cnt) 
        x, y, w, h = cv2.boundingRect(cnt)
        cy = y + (h / 2)

        # Filters: 
        # Area > 500 (ignores small noise)
        # cy > height * 0.4 (ignores the rig hanging from the top)
        # w < width * 0.8 (ignores long horizontal lines from the table edge)
        if area > 500 and cy > (height * 0.4) and w < (width * 0.8): 
            if area > max_area:
                max_area = area
                plant_contour = cnt

    return img_8u, edges, closed_edges, plant_contour

# --- Visualization ---
image_file = r"H:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 5_1\DEV_1AB22C05B465\timelapse_2026-05-07_12-56-08\tl_2026-05-07_12-56-30.tif"

try:
    original, edges, closed_mask, contour = segment_plant_edges(image_file)
    
    result_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if contour is not None:
        cv2.drawContours(result_img, [contour], -1, (0, 0, 255), 3)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Normalized Original')
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Canny Edges')
    axes[1].axis('off')

    axes[2].imshow(closed_mask, cmap='gray')
    axes[2].set_title('Connected Edges (Mask)')
    axes[2].axis('off')

    axes[3].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Final Plant Contour')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error processing image: {e}")