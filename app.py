import cv2
import csv
import pandas as pd

# Step 1: Save the first frame as an image for ROI selection
video = cv2.VideoCapture('C:/Users/thanu/OneDrive/Documents/smart_park/data.mp4')
success, frame = video.read()

if not success:
    print("Error: Unable to read video.")
    exit()

# Save the first frame for ROI selection
cv2.imwrite("frame.jpg", frame)
video.release()
image = cv2.imread("frame.jpg")

# Resize image for usability
height, width, _ = image.shape
image = cv2.resize(image, (int(width * 0.35), int(height * 0.35)))

# Select ROI and save it to CSV
print("Select ROI for parking spaces. Press ENTER after selecting.")
rois = cv2.selectROI("Select Parking Spaces", image, showCrosshair=False, fromCenter=False)
cv2.destroyAllWindows()

# Save ROI to CSV
with open('./rois.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(list(rois))
print(f"ROI saved: {rois}")

# Step 2: Define a function to detect and mark empty/occupied spaces
def check_parking_space(frame, x, y, w, h, low_thresh, high_thresh, min_edges, max_edges):
    sub_img = frame[y:y + h, x:x + w]
    edges = cv2.Canny(sub_img, low_thresh, high_thresh)
    edge_count = cv2.countNonZero(edges)

    # Debug: Output edge count for the ROI
    print(f"ROI ({x}, {y}, {w}, {h}): Edge Count = {edge_count}")

    # Mark as occupied (red) or empty (green) based on edge count
    if min_edges <= edge_count <= max_edges:
        color = (0, 255, 0)  # Green for empty
        status = "Empty"
    else:
        color = (0, 0, 255)  # Red for occupied
        status = "Occupied"
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, status, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Step 3: Setup trackbars for parameter adjustments
cv2.namedWindow("Parameters")
cv2.createTrackbar("Low Threshold", "Parameters", 50, 500, lambda x: None)
cv2.createTrackbar("High Threshold", "Parameters", 150, 500, lambda x: None)
cv2.createTrackbar("Min Edges", "Parameters", 50, 1000, lambda x: None)
cv2.createTrackbar("Max Edges", "Parameters", 300, 1000, lambda x: None)

# Step 4: Load ROI data from CSV
try:
    roi_data = pd.read_csv('./rois.csv', header=None)
    roi_data.columns = ['x', 'y', 'width', 'height']
except FileNotFoundError:
    print("Error: ROI file not found.")
    exit()

# Step 5: Process video and detect parking spaces
video = cv2.VideoCapture('C:/Users/thanu/OneDrive/Documents/smart_park/data.mp4')

while video.isOpened():
    success, frame = video.read()
    if not success:
        print("End of video or cannot read frame.")
        break

    # Resize frame to match ROI size
    height, width, _ = frame.shape
    frame = cv2.resize(frame, (int(width * 0.35), int(height * 0.35)))

    # Get trackbar values for edge detection
    low_thresh = cv2.getTrackbarPos("Low Threshold", "Parameters")
    high_thresh = cv2.getTrackbarPos("High Threshold", "Parameters")
    min_edges = cv2.getTrackbarPos("Min Edges", "Parameters")
    max_edges = cv2.getTrackbarPos("Max Edges", "Parameters")

    # Check parking spaces using ROI data
    for _, roi in roi_data.iterrows():
        check_parking_space(frame, int(roi['x']), int(roi['y']), int(roi['width']), int(roi['height']),
                            low_thresh, high_thresh, min_edges, max_edges)

    # Display the results
    cv2.imshow("Parking Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
