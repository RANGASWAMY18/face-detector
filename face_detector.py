
import cv2
import os

# Step 1: Get the image filename
image_path = input("Enter image name: ").strip().strip('"')

# Step 2: Check whether the image exists
if not os.path.isfile(image_path):
    print("Error: Image file not found.")
    raise SystemExit

# Step 3: Load the image
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to read image.")
    raise SystemExit

# Step 4: Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 5: Load the face detection model
cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Face detector could not be loaded.")
    raise SystemExit

# Step 6: Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# Step 7: Draw a rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

# Step 8: Save the output image
output_path = "detected_faces.jpg"
cv2.imwrite(output_path, image)

# Step 9: Display the result
print("Face detection completed!")
print("Faces detected:", len(faces))
print("Output image:", os.path.abspath(output_path))

cv2.imshow("Face Detector", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
