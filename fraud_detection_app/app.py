from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session
import os
from ultralytics import YOLO
from easyocr import Reader
import zipfile
import pandas as pd
import cv2
from fuzzywuzzy import fuzz, process
import re
import numpy as np
import json
import bcrypt
from flask_sqlalchemy import SQLAlchemy
import matplotlib.pyplot as plt
import seaborn as sns
import io
from flask import Response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
# Set the URI for the database (for SQLite in this case)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'  # For SQLite
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://username:password@localhost/dbname'  # For PostgreSQL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'  # Change this to a strong random key
db = SQLAlchemy(app)


# Load models
classifier = YOLO("C:/Users/ssuri/Documents/aadhaar_fraud_management/infosys/Fraud_Management/classification_model/runs/classify/train/weights/best.pt")  # Replace with your classification model path
detector = YOLO("C:/Users/ssuri/Documents/aadhaar_fraud_management/infosys/Fraud_Management/detection_model/runs/detect/train4/weights/best.pt")  # Replace with your detection model path
reader = Reader(['en'])

class DataResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    srno = db.Column(db.String(50), nullable=False)
    uid = db.Column(db.String(50))
    name = db.Column(db.String(100))
    address = db.Column(db.String(200))
    uid_match_score = db.Column(db.Integer)
    name_match_score = db.Column(db.Integer)
    final_remarks = db.Column(db.String(100))
    document_type = db.Column(db.String(50))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Create the tables
#with app.app_context():
 #   db.create_all()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


@app.route('/register', methods=['GET', 'POST'])  
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return 'Email already exists!', 400

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
       
        return redirect('/login')
   
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])  
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
       
        if user and user.check_password(password):
            session['name'] = user.name
            session['email'] = user.email
            return redirect('/')
        else:
            return 'Invalid email or password!', 401
   
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('name', None)
    session.pop('email', None)
    return redirect('/')

def process_image(image_path):
    if classifier.predict(image_path)[0].probs.numpy().top1 == 0:
        fields = detector(image_path)
        image = cv2.imread(image_path)
        extracted_data = {}
        for field in fields[0].boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = map(int, field[:6])
            field_class = detector.names[class_id]
            cropped_roi = image[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
            text = reader.readtext(gray_roi, detail=0)
            extracted_data[field_class] = ' '.join(text)
        return extracted_data
    return None

# Helper Functions
def normalize_text(text):
    if not text:
        return "text empty"
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split()).lower()

def calculate_match_score(input_value, extracted_value):
    if pd.isna(input_value) or pd.isna(extracted_value):
        return 0
    return fuzz.ratio(str(input_value), str(extracted_value))


def name_match(input_name, extracted_name):
    if extracted_name is None:
        return False
    input_name = normalize_text(input_name)
    extracted_name = normalize_text(extracted_name)

    if input_name == extracted_name:
        return True

    input_parts = input_name.split()
    extracted_parts = extracted_name.split()

    if sorted(input_parts) == sorted(extracted_parts):
        return True

    if len(input_parts) == 2 and len(extracted_parts) == 3:
        if input_parts[0] == extracted_parts[0] and input_parts[1] == extracted_parts[2]:
            return True
    if len(input_parts) == 3 and len(extracted_parts) == 2:
        if extracted_parts[0] == input_parts[0] and extracted_parts[1] == input_parts[2]:
            return True

    for part in input_parts:
        if part not in extracted_parts:
            return False
    return True


def address_match(input_address, extracted_address):
    print(input_address, extracted_address)
    if input_address is None or extracted_address is None:
        return False, 0.0, {}

    # Handle input_address if it's a Series
    if isinstance(input_address, pd.Series):
        input_address = input_address.to_dict()
    
    print(extracted_address)
    extracted_address = normalize_text(extracted_address)
    final_score = 0
    print(extracted_address)
    weights = {
        "State": 0.2,
        "Landmark": 0.2,
        "Premise Building Name": 0.2,
        "City":0.2,
        "Street Road Name":0.1,
        "Floor Number": 0.05,
        "House Flat Number": 0.05
    }
    tokens = extracted_address.split(" ")
    # Component matching logic
    for field, weight in weights.items():
        input_value = input_address.get(field, "")
        match_score = fuzz.token_set_ratio(normalize_text(input_value), extracted_address) if input_value else 0
        input_address[field + " Match Score"] = match_score
        final_score += match_score * weight
    pincode_score = process.extractOne(input_address.get("PINCODE"), tokens)[1]
    input_address['PINCODE Match Score'] = pincode_score
    pincode_matched = True if input_address['PINCODE Match Score'] == 100 else False

    return final_score >= 70 and pincode_matched, final_score, input_address



def compare_data(input_data, json_data):
    excel_data = input_data.copy()
    for idx, row in excel_data.iterrows():
        serial_no = row.get("SrNo")
        uid = row.get("UID")
        extracted = json_data.get(serial_no)

        if extracted:
            extracted_uid = extracted.get("uid", "").replace(" ", "")
            extracted_name = extracted.get("name", "")
            extracted_address = extracted.get("address", "")
            row['Extracted UID'] = extracted_uid
            row['Extracted Name'] = extracted_name
            row['Extracted Address'] = extracted_address
            # UID Match
            uid_match = uid == extracted_uid
            uid_score = 100 if uid_match else 0
            row['UID Match Score'] = uid_score

            # Name Match
            name_match_result = name_match(row.get("Name"), extracted_name)
            name_score = calculate_match_score(row.get("Name"), extracted_name)
            row['Name Match Score'] = name_score
            row['Name Match Percentage'] = name_score

            # Address Match
            address_match_result, address_score, partial_scores = address_match(row, extracted_address)
            if partial_scores:
                row['House Flat Number Match Score'] = partial_scores['House Flat Number Match Score']
                row['Street Road Name Match Score'] = partial_scores['Street Road Name Match Score']
                row['City Match Score'] = partial_scores['City Match Score']
                row['Floor Number Match Score'] = partial_scores['Floor Number Match Score']
                row['Premise Building Name Match Score'] = partial_scores['Premise Building Name Match Score']
                row['Landmark Match Score'] = partial_scores['Landmark Match Score']
                row['State Match Score'] = partial_scores['State Match Score']
                row['Final Address Match'] = address_match_result
                row['Final Address Match Score'] = address_score
                row['PINCODE Match Score'] = partial_scores['PINCODE Match Score']
            
            # Final Match
            overall_match = uid_match and name_match_result and address_match_result

            row['Overall Match'] = overall_match

            if overall_match:
                row['Final Remarks'] = "All matched: UID Matched, Address Matched, Name Matched"
            elif not uid_match:
                row['Final Remarks'] = "UID mismatch, Address Matched, Name Matched"
            elif not name_match_result:
                row['Final Remarks'] = "Name mismatch, UID Matched, Address Matched"
            elif not address_match_result:
                row['Final Remarks'] = "Address mismatch, UID Matched, Name Matched"
            else:
                if extracted_address is None:
                    row['Final Remarks'] = "Address missing in aadhar"
                elif extracted_name is None:
                    row['Final Remarks'] = "Name missing in aadhar"
                else:
                    row['Final Remarks'] = "Non Aadhar"

            row["Document Type"] = "Aadhaar" if overall_match else "Non-Aadhaar"
        else:
            row.replace(float('nan'), 0)
            row['Final Remarks'] = "Non Aadhar"
            row['Document Type'] = "Non Aadhar"
        excel_data.loc[idx] = row


         # Create a new result entry for the database
        result = DataResult(
                srno=serial_no,
                uid=extracted_uid,
                name=extracted_name,
                address=extracted_address,
                uid_match_score=100 if uid == extracted_uid else 0,
                name_match_score=calculate_match_score(row.get("Name"), extracted_name),
                final_remarks="All matched" if uid == extracted_uid and name_match_result else "Non Aadhar",
                document_type="Aadhaar" if uid == extracted_uid and name_match_result else "Non-Aadhaar",
                user_id=1  # Replace with actual user ID if needed
            )
        db.session.add(result) 
    db.session.commit() # Add the result to the session



    return excel_data

@app.route('/download', methods=['GET'])
def download_results():
    file_path = os.path.join(app.config['RESULTS_FOLDER'], 'results.xlsx')
    return send_file(file_path, as_attachment=True)

@app.route('/graph/multiple', methods=['GET'])
def generate_multiple_charts():
    file_path = os.path.join(app.config['RESULTS_FOLDER'], 'results.xlsx')
    if not os.path.exists(file_path):
        return jsonify({"error": "Results file not found."}), 404

    # Load the Excel file
    df = pd.read_excel(file_path)

    # Data preparation
    uid_match_counts = df['UID Match Score'].value_counts()
    doc_type_counts = df['Document Type'].value_counts()
    remarks_counts = df['Final Remarks'].value_counts()
    name_match_counts = df['Name Match Score'].apply(lambda x: 'Match' if x >= 80 else 'Mismatch').value_counts()
    address_match_scores = df['Final Address Match Score'].dropna()

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # UID Match vs Non-Match
    axes[0, 0].pie(
        uid_match_counts,
        labels=uid_match_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['#00b0be', '#ff8ca1']
    )
    axes[0, 0].set_title('UID Match vs Non-Match')

    # Document Type Distribution
    axes[0, 1].bar(
        doc_type_counts.index,
        doc_type_counts.values,
        color=['#ffcd8e', '#59a89c']
    )
    axes[0, 1].set_title('Document Type Distribution')
    axes[0, 1].set_xlabel('Document Type')
    axes[0, 1].set_ylabel('Count')

    # Final Remarks Distribution
    axes[1, 0].bar(
        remarks_counts.index,
        remarks_counts.values,
        color='#8fd787'
    )
    axes[1, 0].set_title('Final Remarks Distribution')
    axes[1, 0].set_xlabel('Remarks')
    axes[1, 0].set_ylabel('Count')

    # Name Match vs Non-Match
    axes[1, 1].pie(
        name_match_counts,
        labels=name_match_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['#ff8ca1', '#f45f74']
    )
    axes[1, 1].set_title('Name Match vs Non-Match')

    # Address Match Score Distribution
    axes[2, 0].hist(
        address_match_scores,
        bins=10,
        color='#e63946'
    )
    axes[2, 0].set_title('Address Match Score Distribution')
    axes[2, 0].set_xlabel('Match Score')
    axes[2, 0].set_ylabel('Frequency')

    # Overall Match Percentage
    overall_match_percentage = df['Overall Match'].value_counts(normalize=True) * 100
    axes[2, 1].pie(
        overall_match_percentage,
        labels=overall_match_percentage.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=['#e63946', '#ff8ca1']
    )
    axes[2, 1].set_title('Overall Match Percentage')

    # Save the combined figure to a BytesIO stream
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return the combined image as a response
    return Response(img.getvalue(), content_type='image/png')

@app.route('/results')
def show_results():
    results = DataResult.query.all()  # Get all results
    return jsonify([{
        'srno': result.srno,
        'uid': result.uid,
        'name': result.name,
        'address': result.address,
        'uid_match_score': result.uid_match_score,
        'name_match_score': result.name_match_score,
        'final_remarks': result.final_remarks,
        'document_type': result.document_type
    } for result in results])

@app.route('/')
def home():
    if 'name' in session:
        return render_template('del.html', name=session['name'])
    return render_template('del.html', name=None)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'zipfile' in request.files and 'excelfile' in request.files:
        zip_file = request.files['zipfile']
        excel_file = request.files['excelfile']

        # Save files
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file.filename)
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        zip_file.save(zip_path)
        excel_file.save(excel_path)

        # Unzip and process images
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(app.config['UPLOAD_FOLDER'])

        image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(('.jpg', '.png'))]
        processed_results = {}

        for image_path in image_paths:
            file_name = os.path.basename(image_path)
            key = file_name.split('.')[0][:3]
            if key not in processed_results:  # Check if key already exists
                extracted_data = process_image(image_path)
                if extracted_data:
                    processed_results[key] = extracted_data

        # Read Excel and compare data
        df = pd.read_excel(excel_path)
        df = df.astype('str')
        comparison_results = compare_data(df, processed_results)
        comparison_results['Accepted/Rejected'] = np.where(comparison_results['Final Remarks'] == 'All matched: UID Matched, Address Matched, Name Matched', 'Accepted', 'Rejected')

        # Save results to a new Excel file
        results_df = pd.DataFrame(comparison_results)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        results_file_path = os.path.join(app.config['RESULTS_FOLDER'], 'results.xlsx')
        results_df.to_excel(results_file_path, index=False)

        return jsonify({"message": "Files processed successfully!", "results": comparison_results[['SrNo', 'Document Type',  'Accepted/Rejected', 'Final Remarks']].to_dict(orient='records')})

    return jsonify({"error": "Both files are required."}), 400

if __name__ == '__main__':
    app.run(debug=True)
