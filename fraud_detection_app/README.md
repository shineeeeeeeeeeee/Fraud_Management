
# Aadhaar Fraud Detection System

This project is an AI-based Aadhaar Fraud Detection system built using Flask. It processes uploaded Aadhaar document images and associated data in Excel format to validate the UID, name, and address details. The system uses YOLO for object detection and EasyOCR for text extraction, alongside fuzzy matching techniques for data validation.

---

## Features

- **Document Validation**: Classifies Aadhaar documents and non-Aadhaar documents using a trained YOLO model.
- **OCR Integration**: Extracts text from images using EasyOCR for validation.
- **Data Matching**: Matches extracted data with input Excel data using fuzzy matching algorithms.
- **SQL Database Integration**: Stores results in an SQLite database for record keeping.
- **Downloadable Results**: Processed results are saved in an Excel file, which can be downloaded.
- **REST API Support**: Provides endpoints for uploading files, viewing results, and downloading the processed file.

---

## Project Directory Structure

The following is the directory structure of the **Aadhaar Fraud Detection** project:

### Directory Descriptions
- **`app.py`**: The main file containing the Flask application logic.
- **`models/`**: Contains the YOLO models used for document detection and classification.
- **`uploads/`**: Temporary storage for uploaded files (e.g., zip and Excel files).
- **`results/`**: Stores processed results like the output Excel files.
- **`static/`**: Holds static files such as CSS, JavaScript, and images.
- **`templates/`**: Contains the HTML templates for rendering web pages.
- **`requirements.txt`**: Lists all the Python dependencies required for the project.
- **`README.md`**: Documentation for the project.

### Note
Make sure to create the directories (e.g., `uploads/` and `results/`) if they do not exist before running the application.


---

## Installation

### Prerequisites

1. Python 3.8 or higher
2. Virtual Environment (optional but recommended)
3. SQLite (installed by default with Python)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/shineeeeeeeeeeee/Fraud-Management.git
   cd Fraud-Management
2. **Set up a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install Dependencies
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
Prepare the Database
```
4. **Prepare the Database**
```bash
flask db init
flask db migrate
flask db upgrade
```

5. **Run the Application**

```bash
python app.py
```
# Usage
1. Access the Web Interface: Open your browser and go to http://127.0.0.1:5000/.

2. Upload Files:

- Upload a zip file containing Aadhaar document images.
- Upload an Excel file containing UID, name, and address details.
3. View Results:

- Check results on the /results endpoint or download the processed results.xlsx file.
4. API Endpoints:

- /upload: Accepts zip and Excel files.
- /results: Returns JSON response of processed results.
- /download: Downloads the results.xlsx file.
# Key Functionalities
1. Image Processing:

- YOLO models classify Aadhaar documents and detect key fields (e.g., UID, name, address).
- EasyOCR extracts text from detected fields.
2. Data Matching:

- Fuzzy matching validates UID, name, and address accuracy.
- Generates scores for each component and provides final validation remarks.
3. Database Storage:

- Stores results in SQLite with User and DataResult models.
4. Excel Output:

- Saves processed results in results.xlsx with columns for extracted and matched data.
# Example Result Columns
- SrNo: Serial number of the input data.
- UID Match Score: Score for UID match.
- Name Match Score: Score for name match.
- Final Remarks: Overall validation result (e.g., "All matched", "UID mismatch").
- Accepted/Rejected: Whether the document was accepted or rejected.
# Technologies Used
- Flask: Web framework for building the application.
- YOLO: Used for object detection and classification.
- EasyOCR: Extracts text from Aadhaar images.
- SQLAlchemy: Database ORM for managing results.
- FuzzyWuzzy: String matching for validation.
- OpenCV: Image preprocessing.
# Dependencies
The dependencies are listed in requirements.txt. Install them using:

```bash
pip install -r requirements.txt
```
# Contributing
1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes and open a pull request.
# Contact
For any queries, feel free to reach out at:

- Email: your-email@example.com
- GitHub: your-username
# License
This project is licensed under the MIT License.
