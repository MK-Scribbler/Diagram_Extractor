# Diagram Scrapper

This project is a Flask web application for extracting, previewing, cropping, and cleaning diagrams/images from a web URL, PDF file, or local images. It uses a YOLO model for diagram detection and supports cropping and cleaning workflows for dataset creation.

---

## Project Structure

```
img-scrapping-flask/
├── app.py                # Main Flask application
├── best.pt               # YOLO model weights
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── results/              # Output/results directory for cleaned images
├── static/
│   ├── diagrams_web_filtered/      # Images filtered from web scraping
│   ├── diagrams_pdf_filtered/      # Images filtered from PDF
│   └── diagrams_folder_filtered/   # Images uploaded for cleaning
├── templates/
│   ├── index.html        # Home/upload page
│   └── preview.html      # Image preview/cropping/selection page
├── uploads/              # Uploaded files (PDFs, images)
├── venv/                 # Python virtual environment
```

---

## Features

- **Extract images from a web URL** (with Selenium/Playwright support for dynamic sites).
- **Extract diagrams from PDF files** (requires Poppler).
- **Upload and clean local images** (multi-select supported).
- **Preview and crop images** with Cropper.js before downloading.
- **Download selected/cropped images** for dataset creation.
- **Efficient cropping:** Cropped images are resized and compressed in-browser before upload to avoid large POST requests.

---

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd img-scrapping-flask
   ```

2. **Create and activate a virtual environment:**
   - On Windows:
     ```sh
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **(For PDF support) Install Poppler:**
   - Download Poppler for Windows from [Poppler Windows releases](https://github.com/oschwartz10612/poppler-windows/releases/).
   - Extract and set the `POPPLER_PATH` in `app.py` to the `bin` folder inside the extracted directory.

5. **Place your YOLO model weights (`best.pt`) in the project root.**

---

## Usage

1. **Start the Flask application:**
   ```sh
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://127.0.0.1:8080
   ```

3. **Choose to:**
   - Enter a web URL to extract diagrams from images on a webpage,
   - Upload a PDF file to extract diagrams from its pages, or
   - Select and upload multiple images for cropping/cleaning.

4. **Preview, crop, and select the images you want to keep.**

5. **Download the selected/cropped images for your dataset.**

---

## Dependencies

- Flask
- Requests
- BeautifulSoup4
- pdf2image
- OpenCV (cv2)
- numpy
- ultralytics
- shutil (standard library)
- Poppler (external, for PDF support)
- Cropper.js (frontend)
- Selenium or Playwright (for advanced web scraping, optional)

Install all Python dependencies using:
```sh
pip install -r requirements.txt
```

---

## Notes

- Cropped images are resized and compressed in-browser to avoid large uploads and 413 errors.
- If you encounter "Request Entity Too Large", try cropping smaller areas or fewer images.
- For PDF extraction, Poppler must be installed and the path set correctly in `app.py`.
- The application creates and manages folders for uploads and filtered images automatically.
- For web scraping, Playwright is recommended for dynamic sites (see documentation for usage).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
