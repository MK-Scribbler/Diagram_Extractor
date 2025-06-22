# img-scrapping-flask

This project is a Flask web application that allows users to extract diagrams/images from a web URL or PDF file, preview the results, select images, and download them as a ZIP file. It uses a YOLO model for diagram detection and supports both web and PDF sources.

---

## Project Structure

```
img-scrapping-flask/
├── app.py                # Main Flask application
├── best.pt               # YOLO model weights
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── results/              # (Optional) Output/results directory
├── runs/                 # (Optional) YOLO runs directory
├── static/               # Static files (filtered images, etc.)
├── templates/
│   ├── index.html        # Home/upload page
│   └── preview.html      # Image preview/selection page
├── uploads/              # Uploaded files (PDFs, images)
├── venv/                 # Python virtual environment
```

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
   http://127.0.0.1:5000
   ```

3. **Choose to either:**
   - Enter a web URL to extract diagrams from images on a webpage, or
   - Upload a PDF file to extract diagrams from its pages.

4. **Preview the detected diagrams/images. Select the ones you want to download.**

5. **Download the selected images as a ZIP file.**

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

Install all Python dependencies using:
```sh
pip install -r requirements.txt
```

---

## Notes

- Make sure you have the YOLO model weights (`best.pt`) in the project directory.
- For PDF extraction, Poppler must be installed and the path set correctly in `app.py`.
- The application creates and manages folders for uploads and filtered images automatically.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.