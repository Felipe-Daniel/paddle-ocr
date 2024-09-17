import os
import cv2
from matplotlib import pyplot as plt
from paddleocr import PaddleOCR
from paddleocr.tools.infer.utility import draw_ocr
from pdf2image import convert_from_path

# Initialize PaddleOCR
ocr = PaddleOCR(lang='pt')

# Path to the PDF file
pdf_path = 'CNH.pdf'

# Convert PDF to images
images = convert_from_path(pdf_path, fmt='jpeg')

# Loop through images and apply OCR
for page_number, img in enumerate(images):
    # Save each page as JPEG for clarity (optional)
    img_path = f'page_{page_number + 1}.jpeg'
    img.save(img_path, 'JPEG')

    # Perform OCR on the image
    result = ocr.ocr(img_path)
    result = result[0]

    # # Print OCR results
    # for line in result:
    #     print(line)

    # Extracting detected components
    boxes = [line[0] for line in result]  # Get bounding boxes
    texts = [line[1][0] for line in result]  # Get recognized texts
    scores = [line[1][1] for line in result]  # Get confidence scores

    # Specifying font path for draw_ocr method
    font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')

    # Load the image
    img = cv2.imread(img_path)

    # Convert the image to RGB (OpenCV loads images in BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Visualize the image with annotations
    plt.figure(figsize=(15, 15))

    # Draw the OCR annotations on the image
    annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)

    # Display the image with annotations using matplotlib
    plt.imshow(annotated)
    plt.show()
