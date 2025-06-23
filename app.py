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
from tempfile import mkdtemp
from ultralytics import YOLO
import time

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

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    total_height = driver.execute_script("return document.body.scrollHeight")
    for pos in range(0, total_height, 800):
        driver.execute_script(f"window.scrollTo(0, {pos});")
        time.sleep(1)

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img")

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

    saved_boxes = []
    count = 1
    tmp_dir = mkdtemp()

    for img_url in image_urls:
        try:
            img_data = requests.get(img_url, timeout=5).content
            ext = os.path.splitext(urlparse(img_url).path)[-1] or ".jpg"
            img_path = os.path.join(tmp_dir, f"img_{count}{ext}")
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
                    cropped = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
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
    for relative_path in selected:
        src = os.path.join('static', relative_path)
        ext = os.path.splitext(relative_path)[-1]  # Preserve original extension (e.g., .jpg, .png)
        dst = os.path.join(extracted_dir, f"{next_number}{ext}")

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
