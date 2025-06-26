import os
import cv2
import shutil
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template, redirect, url_for
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tempfile import mkdtemp
from PIL import Image
import io
from ultralytics import YOLO
import time
from pdf2image import convert_from_path
import base64

app = Flask(__name__)

# === Config ===
UPLOAD_FOLDER = 'uploads'
WEB_OUTPUT_FOLDER = 'static/diagrams_web_filtered'
PDF_OUTPUT_FOLDER = 'static/diagrams_pdf_filtered'
POPPLER_PATH = r"C:\\poppler\\poppler-24.08.0\\Library\\bin"
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.8
PADDING = 25

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WEB_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

@app.template_filter('basename')
def basename_filter(value):
    return os.path.basename(value)

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


@app.route('/', methods=['POST'])
def scrape_images():
    url = request.form.get('url')
    if not url:
        return redirect(url_for('index'))

    shutil.rmtree(WEB_OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(WEB_OUTPUT_FOLDER, exist_ok=True)

    # Headless browser setup
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    # Scroll to load lazy images
    total_height = driver.execute_script("return document.body.scrollHeight")
    for pos in range(0, total_height, 800):
        driver.execute_script(f"window.scrollTo(0, {pos});")
        time.sleep(1)

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img")

    # Best image URL resolver
    def get_best_src(tag):
        srcset = tag.get("srcset")
        if srcset:
            items = [item.strip().split() for item in srcset.split(",") if item.strip()]
            best = max(items, key=lambda x: int(x[-1].rstrip('wｘx')) if x[-1][:-1].isdigit() else 0)
            return best[0]
        return tag.get("data-src") or tag.get("src")

    image_urls = []
    for tag in img_tags:
        src = get_best_src(tag)
        if src:
            full_url = urljoin(url, src.strip())
            image_urls.append(full_url)

    count = 1
    for img_url in image_urls:
        try:
            img_data = requests.get(img_url, timeout=5).content
            ext = os.path.splitext(urlparse(img_url).path)[-1] or ".jpg"
            out_path = os.path.join(WEB_OUTPUT_FOLDER, f"{count}{ext}")
            with open(out_path, "wb") as f:
                f.write(img_data)
            count += 1
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")

    # Preview all saved images
    image_urls = [url_for('static', filename=f'diagrams_web_filtered/{img}') for img in sorted(os.listdir(WEB_OUTPUT_FOLDER))]
    return render_template('preview.html', image_urls=image_urls)


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    pdf_file = request.files.get('pdf')
    if not pdf_file:
        return redirect(url_for('index'))

    saved_boxes = []
    count = 1

    shutil.rmtree(PDF_OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    for idx, page in enumerate(pages):
        page_path = os.path.join(UPLOAD_FOLDER, f"page_{idx+1}.jpg")
        page.save(page_path)
        image = cv2.imread(page_path)
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
            if not result.boxes or len(result.boxes) == 0:
                continue

            conf = float(result.boxes[0].conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            this_box = (x1, y1, x2, y2)
            if any(iou(this_box, b) > IOU_THRESHOLD for b in saved_boxes):
                continue

            out_path = os.path.join(PDF_OUTPUT_FOLDER, f"{count}.jpg")
            cv2.imwrite(out_path, cropped)
            saved_boxes.append(this_box)
            count += 1

        os.remove(page_path)

    image_urls = [url_for('static', filename=f'diagrams_pdf_filtered/{img}') for img in sorted(os.listdir(PDF_OUTPUT_FOLDER))]
    return render_template('preview.html', image_urls=image_urls)


@app.route('/download_selected', methods=['POST'])
def download_selected():
    selected = request.form.getlist('selected_images')
    if not selected:
        return render_template("message.html", status="error", message="⚠️ No images selected for download.", redirect_url=url_for('index'))

    extracted_dir = os.path.join('results', 'Diagrams_Extracted')
    os.makedirs(extracted_dir, exist_ok=True)

    # Step 1: Find the last used number in the folder
    existing_numbers = []
    for filename in os.listdir(extracted_dir):
        name, ext = os.path.splitext(filename)
        if name.isdigit():
            existing_numbers.append(int(name))
    next_number = max(existing_numbers, default=0) + 1

    # Step 2: Copy images with new names in sequential order
    added = False
    for idx, relative_path in enumerate(selected):
        src = os.path.join('static', relative_path)
        ext = os.path.splitext(relative_path)[-1]  # Preserve original extension (e.g., .jpg, .png)
        dst = os.path.join(extracted_dir, f"{next_number}{ext}")

        # Check for cropped image data
        cropped_key = f"cropped_image_data_{idx}"
        cropped_data = request.form.get(cropped_key)
        if cropped_data and cropped_data.startswith("data:image"):
            # Save the cropped image
            header, encoded = cropped_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            with open(dst, "wb") as f:
                f.write(img_bytes)
            added = True
            next_number += 1
            continue

        # Otherwise, copy the original
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            added = True
            next_number += 1

    if not added:
        return render_template("message.html", status="error", message="⚠️ No valid images found to copy.", redirect_url=url_for('index'))

    return render_template("message.html", status="success", message="✅ Selected images saved in order!", redirect_url=url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
