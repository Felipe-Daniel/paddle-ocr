from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import cv2 #opencv
import os # folder directory navigation


# Setup model
ocr_model = PaddleOCR(lang='en')
img_path = os.path.join('.', 'drug1.jpg')
result = ocr_model.ocr(img_path)
for res in result:
    print(res[1][0])

# Extracting detected components
boxes = [res[0] for res in result] #
texts = [res[1][0] for res in result]
scores = [res[1][1] for res in result]

# Specifying font path for draw_ocr method
font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')

# Import our image - drug 1/2/3
# imports image
img = cv2.imread(img_path)

# reorders the color channels
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Visualize our image and detections
# resizing display area
plt.figure(figsize=(15,15))

# draw annotations on image
annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)

# show the image using matplotlib
plt.imshow(annotated)