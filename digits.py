import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

print("Loading trained CNN model...")

model = tf.keras.models.load_model("digit_model.keras")

print("Model loaded successfully.")

digit_names = [
    "ZERO",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE"
]

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select Handwritten Digit Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
        ("JPG Files", "*.jpg"),
        ("JPEG Files", "*.jpeg"),
        ("PNG Files", "*.png"),
        ("All Files", "*.*")
    ]
)

root.destroy()

if not image_path:
    print("\nNo image selected.")
    exit()

print("\nSelected image:")
print(image_path)

image = cv2.imread(image_path)

if image is None:
    print("\nERROR: Image could not be loaded.")
    exit()

original = image.copy()

# Detect coloured ink of any hue (orange, blue, red, green, purple, etc.).
# Only this colour-detection section is changed; the rest of your code is unchanged.
hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

lower_colour = np.array([
    0,
    35,
    20
])

upper_colour = np.array([
    179,
    255,
    255
])

blue_mask = cv2.inRange(
    hsv,
    lower_colour,
    upper_colour
)

kernel = np.ones(
    (3, 3),
    np.uint8
)

blue_mask = cv2.morphologyEx(
    blue_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

blue_mask = cv2.morphologyEx(
    blue_mask,
    cv2.MORPH_OPEN,
    kernel
)

contours, _ = cv2.findContours(
    blue_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

valid_contours = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)

    if area > 100 and w > 15 and h > 30:
        aspect_ratio = w / float(h)

        if 0.15 < aspect_ratio < 3.5:
            valid_contours.append(contour)

if len(valid_contours) == 0:
    print("\nNo handwritten digit detected.")
    exit()

all_points = []

for contour in valid_contours:
    points = contour.reshape(-1, 2)
    all_points.extend(points)

all_points = np.array(all_points)

x_min = int(
    np.min(all_points[:, 0])
)

y_min = int(
    np.min(all_points[:, 1])
)

x_max = int(
    np.max(all_points[:, 0])
)

y_max = int(
    np.max(all_points[:, 1])
)

padding = 25

x_start = max(
    0,
    x_min - padding
)

y_start = max(
    0,
    y_min - padding
)

x_end = min(
    image.shape[1],
    x_max + padding
)

y_end = min(
    image.shape[0],
    y_max + padding
)

digit = blue_mask[
    y_start:y_end,
    x_start:x_end
]

digit_h, digit_w = digit.shape

size = max(
    digit_w,
    digit_h
)

square = np.zeros(
    (size, size),
    dtype=np.uint8
)

offset_x = (
    size - digit_w
) // 2

offset_y = (
    size - digit_h
) // 2

square[
    offset_y:offset_y + digit_h,
    offset_x:offset_x + digit_w
] = digit

digit = cv2.resize(
    square,
    (20, 20),
    interpolation=cv2.INTER_AREA
)

canvas = np.zeros(
    (28, 28),
    dtype=np.uint8
)

canvas[
    4:24,
    4:24
] = digit

ys, xs = np.where(
    canvas > 0
)

if len(xs) > 0:
    center_x = (
        xs.min() + xs.max()
    ) / 2

    center_y = (
        ys.min() + ys.max()
    ) / 2

    shift_x = int(
        14 - center_x
    )

    shift_y = int(
        14 - center_y
    )

    transformation_matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    canvas = cv2.warpAffine(
        canvas,
        transformation_matrix,
        (28, 28)
    )

digit = canvas.astype(
    "float32"
) / 255.0

digit = digit.reshape(
    1,
    28,
    28,
    1
)

prediction = model.predict(
    digit,
    verbose=0
)

predicted_digit = int(
    np.argmax(prediction[0])
)

confidence = float(
    prediction[0][predicted_digit] * 100
)

name = digit_names[
    predicted_digit
]

print("\nPREDICTION RESULTS")
print("------------------")
print(
    "Predicted Digit:",
    predicted_digit
)
print(
    "Digit Name:",
    name
)
print(
    "Confidence:",
    round(confidence, 2),
    "%"
)

print("\nALL PREDICTIONS")
print("------------------")

for i in range(10):
    percentage = float(
        prediction[0][i] * 100
    )

    print(
        f"{digit_names[i]:8s}: {percentage:.2f}%"
    )

display = original.copy()

cv2.rectangle(
    display,
    (x_start, y_start),
    (x_end, y_end),
    (0, 255, 0),
    4
)

label = (
    f"{name} ({predicted_digit}) "
    f"{confidence:.2f}%"
)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2

(tw, th), baseline = cv2.getTextSize(
    label,
    font,
    font_scale,
    thickness
)

label_x = x_start

label_y = y_start - 15

if label_y < th + 15:
    label_y = y_start + th + 15

cv2.rectangle(
    display,
    (
        label_x,
        label_y - th - 10
    ),
    (
        label_x + tw + 15,
        label_y + 5
    ),
    (0, 255, 0),
    -1
)

cv2.putText(
    display,
    label,
    (
        label_x + 7,
        label_y
    ),
    font,
    font_scale,
    (0, 0, 0),
    thickness,
    cv2.LINE_AA
)

scale = 3

display = cv2.resize(
    display,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

cv2.namedWindow(
    "Handwritten Digit Prediction",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Handwritten Digit Prediction",
    900,
    700
)

cv2.imshow(
    "Handwritten Digit Prediction",
    display
)

cv2.waitKey(0)

cv2.destroyAllWindows()