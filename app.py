import os
import io
import cv2
import numpy as np
import shutil
from flask import Flask, request, render_template, redirect, url_for,send_file
from pdf2image import convert_from_path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from ultralytics import YOLO
import requests
from tempfile import mkdtemp
import zipfile

app = Flask(__name__)

# === Config ===
UPLOAD_FOLDER = 'uploads'
WEB_OUTPUT_FOLDER = 'static/diagrams_web_filtered'
PDF_OUTPUT_FOLDER = 'static/diagrams_pdf_filtered'
MODEL_PATH = "runs/detect/train/weights/best.pt"
POPPLER_PATH = r"C:\\poppler\\poppler-24.08.0\\Library\\bin"
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.8
PADDING = 25

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WEB_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf' not in request.files:
        return "No file part", 400

    file = request.files['pdf']
    if file.filename == '':
        return "No selected file", 400

    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)

    shutil.rmtree(PDF_OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)

    saved_boxes = []
    count = 1
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)

    for idx, page in enumerate(pages):
        page_img_path = os.path.join(UPLOAD_FOLDER, f"page_{idx+1}.jpg")
        page.save(page_img_path)
        image = cv2.imread(page_img_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 60000 or area > 0.9 * image.shape[0] * image.shape[1]:
                continue

            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(image.shape[1], x + w + PADDING)
            y2 = min(image.shape[0], y + h + PADDING)
            cropped = image[y1:y2, x1:x2]

            result = model.predict(cropped, verbose=False)[0]
            if not result.boxes or float(result.boxes[0].conf[0]) < CONFIDENCE_THRESHOLD:
                continue

            this_box = (x1, y1, x2, y2)
            if any(iou(this_box, sb) > IOU_THRESHOLD for sb in saved_boxes):
                continue

            out_path = os.path.join(PDF_OUTPUT_FOLDER, f"{count}.jpg")
            cv2.imwrite(out_path, cropped)
            saved_boxes.append(this_box)
            count += 1

        os.remove(page_img_path)

    image_urls = [url_for('static', filename=f'diagrams_pdf_filtered/{img}') for img in sorted(os.listdir(PDF_OUTPUT_FOLDER))]
    return render_template('preview.html', image_urls=image_urls)

@app.route('/', methods=['POST'])
def scrape_images():
    url = request.form.get('url')
    if not url:
        return redirect(url_for('index'))

    shutil.rmtree(WEB_OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(WEB_OUTPUT_FOLDER, exist_ok=True)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("img")
    image_urls = [urljoin(url, tag.get("src")) for tag in img_tags if tag.get("src")]

    saved_boxes = []
    count = 1
    tmp_dir = mkdtemp()

    for img_url in image_urls:
        try:
            img_data = requests.get(img_url, timeout=5).content
            img_path = os.path.join(tmp_dir, os.path.basename(urlparse(img_url).path))
            with open(img_path, "wb") as f:
                f.write(img_data)
            image = cv2.imread(img_path)
            if image is None:
                continue

            result = model.predict(image, verbose=False)[0]
            if not result.boxes:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                this_box = (x1, y1, x2, y2)
                if all(iou(this_box, b) < IOU_THRESHOLD for b in saved_boxes):
                    cropped = image[y1:y2, x1:x2]
                    out_path = os.path.join(WEB_OUTPUT_FOLDER, f"{count}.jpg")
                    cv2.imwrite(out_path, cropped)
                    saved_boxes.append(this_box)
                    count += 1
        except Exception as e:
            print(f"Error processing {img_url}: {e}")

    shutil.rmtree(tmp_dir)
    image_urls = [url_for('static', filename=f'diagrams_web_filtered/{img}') for img in sorted(os.listdir(WEB_OUTPUT_FOLDER))]
    return render_template('preview.html', image_urls=image_urls)

@app.route('/download_selected', methods=['POST'])
def download_selected():
    selected = request.form.getlist('selected_images')
    if not selected:
        return "No images selected for download.", 400

    # Create an in-memory zip archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for image_path in selected:
            # Resolve path safely
            filename = os.path.basename(image_path)
            full_path = os.path.join("static", "images", filename)
            if os.path.exists(full_path):
                zip_file.write(full_path, arcname=filename)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='selected_diagrams.zip'
    )

if __name__ == '__main__':
    app.run(debug=True)
