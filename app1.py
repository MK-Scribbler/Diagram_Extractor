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
from playwright.sync_api import sync_playwright

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024* 1024 * 1024 * 1024  # 200MB, adjust as needed

# === Config ===
UPLOAD_FOLDER = 'uploads'
WEB_OUTPUT_FOLDER = 'static/diagrams_web_filtered'
PDF_OUTPUT_FOLDER = 'static/diagrams_pdf_filtered'
POPPLER_PATH = r"C:\\poppler\\poppler-24.08.0\\Library\\bin"
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.8
PADDING = 25

import os
from openpyxl import Workbook, load_workbook

def log_image_to_excel(image_name, source, excel_path='image_log.xlsx'):
    # Only log if the image exists in Diagrams_Extracted
    results_dir = r'E:\Navi-Learn\img-scrapping-flask\results\Diagrams_Extracted'
    image_path = os.path.join(results_dir, image_name)
    if not os.path.exists(image_path):
        return  # Do not log if the image isn't present

    # If file doesn't exist, create it with headers
    if not os.path.exists(excel_path):
        wb = Workbook()
        ws = wb.active
        ws.append(['Image Name', 'Source'])
        wb.save(excel_path)

    # Load existing workbook
    wb = load_workbook(excel_path)
    ws = wb.active

    # Check for duplicate image name
    existing_names = set()
    for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
        existing_names.add(row[0])
    if image_name in existing_names:
        return  # Skip duplicate

    # Append new row
    ws.append([image_name, source])
    wb.save(excel_path)




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
    # Fix: Ensure URL has protocol
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'https://' + url

    shutil.rmtree(WEB_OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(WEB_OUTPUT_FOLDER, exist_ok=True)

    # === ADD THIS: Create sources list ===
    sources = []

    from playwright.sync_api import sync_playwright
    import time

    logo_keywords = ['logo', 'icon', 'sprite', 'favicon', 'avatar', 'profile', 'thumb', 'placeholder']

    def is_google_images(url):
        return "tbm=isch" in url or "google.com/search" in url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

        # ── allow slow pages ────────────────────────────────────────────────────
        page.set_default_navigation_timeout(0)   # never abort .goto()
        page.set_default_timeout(60_000)         # waits/selectors up to 60 s each

        # ── robust navigation with graceful fallback ───────────────────────────
        try:
            page.goto(url, wait_until="domcontentloaded")   # faster than full "load"
        except Exception as nav_err:
            print(f"[WARN] first navigation timed‑out → retrying: {nav_err}")
            try:
                # very short wait‑condition: just wait for the request to commit
                page.goto(url, wait_until="commit")
            except Exception as second_err:
                # total failure → close browser and show friendly message
                browser.close()
                return render_template(
                    "message.html",
                    status="error",
                    message="⚠️ Could not load that page",
                    redirect_url=url_for('index')
                )

        # page loaded ⇒ continue exactly as before
        try:
            page.wait_for_selector("img",
                                state="attached",
                                timeout=20000)    # keep this long wait for images
        except Exception:
            print("Warning: No <img> tag found or took too long to load.")
        time.sleep(2)

        # Scroll to load more images
        for _ in range(5):
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(800)

        count = 1

        if is_google_images(url):
            # --- Google Images logic: click thumbnails for high-res ---
            thumbnails = page.query_selector_all("img.Q4LuWd, img")
            saved_srcs = set()
            for thumb in thumbnails:
                try:
                    thumb.scroll_into_view_if_needed()
                    thumb.click()
                    page.wait_for_timeout(800)
                    images = page.query_selector_all("img.n3VNCb, img")
                    for img in images:
                        src = img.get_attribute("src")
                        if not src:
                            continue
                        src = urljoin(url, src)
                        if not src.lower().startswith("http"):
                            continue
                        if any(word in src.lower() for word in logo_keywords):
                            continue
                        if src in saved_srcs:
                            continue  # Skip if already saved
                        try:
                            img_data = requests.get(src, timeout=5).content
                            img_pil = Image.open(io.BytesIO(img_data))
                            w, h = img_pil.size
                            if w < 80 or h < 80:
                                continue
                            ext = os.path.splitext(urlparse(src).path)[-1] or ".jpg"
                            out_path = os.path.join(WEB_OUTPUT_FOLDER, f"{count}{ext}")
                            with open(out_path, "wb") as f:
                                f.write(img_data)
                            saved_srcs.add(src)
                            # === ADD THIS: Append source ===
                            sources.append(src)
                            count += 1
                            break  # Only save the first valid high-res image per thumbnail
                        except Exception:
                            continue
                except Exception as e:
                    print(f"Error processing thumbnail: {e}")
        else:
            # --- General website logic: just collect all <img> tags ---
            images = page.query_selector_all("img")
            for img in images:
                src = img.get_attribute("src")
                if not src:
                    continue
                src = urljoin(url, src)  # Resolve relative URLs to absolute
                if not src.lower().startswith("http"):
                    continue
                if any(word in src.lower() for word in logo_keywords):
                    continue
                try:
                    img_data = requests.get(src, timeout=5).content
                    img_pil = Image.open(io.BytesIO(img_data))
                    w, h = img_pil.size
                    if w < 80 or h < 80:
                        continue
                    ext = os.path.splitext(urlparse(src).path)[-1] or ".jpg"
                    out_path = os.path.join(WEB_OUTPUT_FOLDER, f"{count}{ext}")
                    with open(out_path, "wb") as f:
                        f.write(img_data)
                    # === ADD THIS: Append source ===
                    sources.append(src)
                    count += 1
                except Exception:
                    continue
        browser.close()
# Preview all saved images
    saved_files = sorted(os.listdir(WEB_OUTPUT_FOLDER))
    if not saved_files:
        return render_template(
            "message.html",
            status="error",
            message="⚠️ No valid images were found on the page.",
            redirect_url=url_for("index"),
            redirect_message="Try another website"
        )

    image_urls = [url_for('static', filename=f'diagrams_web_filtered/{img}') for img in saved_files]
    return render_template('preview.html', image_urls=image_urls, sources=sources)





from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return render_template("message.html", status="error",
        message="⚠️ The file is too large. Maximum allowed size is 1GB.",
        redirect_url=url_for('index')), 413

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    import fitz  # PyMuPDF

    pdf_file = request.files.get('pdf')
    if not pdf_file:
        return redirect(url_for('index'))

    shutil.rmtree(PDF_OUTPUT_FOLDER, ignore_errors=True)
    os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    pdf_file.save(pdf_path)

    doc = fitz.open(pdf_path)
    count = 1

    # === ADD THIS: Create sources list ===
    sources = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            out_path = os.path.join(PDF_OUTPUT_FOLDER, f"page{page_num+1}_img{img_index+1}.{image_ext}")
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            # === ADD THIS: Append PDF source string ===
            pdf_source = f"{pdf_file.filename}_page{page_num+1}_img{img_index+1}"
            sources.append(pdf_source)
            count += 1

    image_urls = [url_for('static', filename=f'diagrams_pdf_filtered/{img}') for img in sorted(os.listdir(PDF_OUTPUT_FOLDER))]
    # === ENSURE sources is passed to template ===
    return render_template('preview.html', image_urls=image_urls, sources=sources)



@app.route('/download_selected', methods=['POST'])
def download_selected():
    selected = request.form.getlist('selected_images')
    if not selected:
        return render_template("message.html", status="error", message="⚠️ No images selected for download.", redirect_url=url_for('index'))

    # If coming from data clean block, save to E:\Cleaned data with original names
    if request.form.get('clean_local'):
        extracted_dir = r'E:\Cleaned data'
        #extracted_dir = r'E:\sample'
        os.makedirs(extracted_dir, exist_ok=True)
        added = False
        for idx, relative_path in enumerate(selected):
            src = os.path.join('static', relative_path)
            original_name = os.path.basename(relative_path)
            dst = os.path.join(extracted_dir, original_name)

            cropped_key = f"cropped_image_data_{idx}"
            cropped_data = request.form.get(cropped_key)
            if cropped_data and cropped_data.startswith("data:image"):
                header, encoded = cropped_data.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                with open(dst, "wb") as f:
                    f.write(img_bytes)
                added = True
                continue

            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                added = True

        if not added:
            return render_template("message.html", status="error", message="⚠️ No valid images found to copy.", redirect_url=url_for('index'))

        return render_template("message.html", status="success", message="✅ Selected images saved with original names in Cleaned data!", redirect_url=url_for('index'))

    # --- Default behavior for web/pdf ---
    extracted_dir = os.path.join('results', 'Diagrams_Extracted')
    os.makedirs(extracted_dir, exist_ok=True)

    # Step 1: Find the last used number in the folder
    existing_numbers = []
    for filename in os.listdir(extracted_dir):
        name, ext = os.path.splitext(filename)
        if name.isdigit():
            existing_numbers.append(int(name))
    next_number = max(existing_numbers, default=19999) + 1

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
            source = request.form.get(f'source_{idx}', '')
            log_image_to_excel(os.path.basename(dst), source)

            next_number += 1
            continue

        # Otherwise, copy the original
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            added = True
            source = request.form.get(f'source_{idx}', '')
            log_image_to_excel(os.path.basename(dst), source)
            next_number += 1

    if not added:
        return render_template("message.html", status="error", message="⚠️ No valid images found to copy.", redirect_url=url_for('index'))

    return render_template("message.html", status="success", message="✅ Selected images saved in order!", redirect_url=url_for('index'))

@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    files = request.files.getlist('images')
    if not files:
        return redirect(url_for('index'))

    FOLDER_UPLOAD_OUTPUT = 'static/diagrams_folder_filtered'
    shutil.rmtree(FOLDER_UPLOAD_OUTPUT, ignore_errors=True)
    os.makedirs(FOLDER_UPLOAD_OUTPUT, exist_ok=True)

    for file in files:
        filename = file.filename
        ext = os.path.splitext(filename)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            out_path = os.path.join(FOLDER_UPLOAD_OUTPUT, filename)
            file.save(out_path)

    image_urls = [url_for('static', filename=f'diagrams_folder_filtered/{img}') for img in sorted(os.listdir(FOLDER_UPLOAD_OUTPUT))]
    # Pass clean_local=True to preview.html
    return render_template('preview.html', image_urls=image_urls, clean_local=True,sources=[])

@app.route('/save_cleaned_image', methods=['POST'])
def save_cleaned_image():
    import base64
    import os

    data = request.json
    filename = data.get('filename')
    cropped_data = data.get('cropped_data')

    if not filename or not cropped_data:
        return {"status": "error", "message": "Missing data"}, 400

    extracted_dir = r'E:\Cleaned data'
    os.makedirs(extracted_dir, exist_ok=True)
    dst = os.path.join(extracted_dir, filename)

    if cropped_data.startswith("data:image"):
        header, encoded = cropped_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        with open(dst, "wb") as f:
            f.write(img_bytes)
        return {"status": "success", "message": "Image saved"}
    else:
        return {"status": "error", "message": "Invalid image data"}, 400

@app.route('/delete_cleaned_image', methods=['POST'])
def delete_cleaned_image():
    import os
    data = request.json
    filename = data.get('filename')
    if not filename:
        return {"status": "error", "message": "Missing filename"}, 400

    extracted_dir = r'E:\Cleaned data'
    file_path = os.path.join(extracted_dir, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return {"status": "success", "message": "File deleted"}
        except Exception as e:
            return {"status": "error", "message": str(e)}, 500
    else:
        return {"status": "not_found", "message": "File does not exist"}, 200

if __name__ == '__main__':
    app.run(debug=True,port=8080)
