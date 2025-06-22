# img-scrapping-flask

This project is a Flask web application that allows users to scrape images from a specified URL or PDF, preview the extracted images, remove unwanted images, and download the selected images.

## Project Structure

```
img-scrapping-flask
├── app.py
├── README.md
├── requirements.txt
├── results/
├── runs/
├── static/
├── templates/
│   ├── index.html
│   └── preview.html
├── train/
├── uploads/
├── valid/
├── venv/
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd img-scrapping-flask
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

5. **(For PDF support) Install Poppler:**
   - Download Poppler for Windows from [Poppler Windows releases](https://github.com/oschwartz10612/poppler-windows/releases/).
   - Extract and set the `POPPLER_PATH` in `app.py` to the `bin` folder inside the extracted directory.

## Usage

1. **Run the Flask application:**
   ```
   python app.py
   ```

2. **Open your web browser and navigate to:**
   ```
   http://127.0.0.1:5000
   ```

3. **Input the URL or upload a PDF from which you want to scrape images and submit the form.**

4. **Preview the extracted images on the next page. You can select which images to keep or remove.**

5. **Download the selected images.**

## Dependencies

- Flask
- Requests
- BeautifulSoup4
- pdf2image
- OpenCV (cv2)
- numpy
- ultralytics
- shutil
- Poppler (external, for PDF support)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.