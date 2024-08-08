import cv2
import numpy as np

# Create a blank image
image = np.zeros((500, 500, 3), dtype="uint8")

# Display the image in a window
cv2.imshow("Test Window", image)

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
