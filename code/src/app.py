import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import extract_msg
from loan_banking_processor import LoanBankingEmailProcessor

app = Flask(__name__)
processor = LoanBankingEmailProcessor()

@app.route('/process_emails', methods=['POST'])
def process_emails():
    try:
        data = request.json
        if not data or 'emails' not in data:
            return jsonify({"error": "Invalid JSON: 'emails' key missing"}), 400
        emails = data.get('emails', [])
        results = []
        for email in emails:
            category, subcategory, confidence = processor.categorize_email(email)
            is_duplicate = processor.detect_duplicate(email)
            results.append({
                'subject': email['subject'],
                'from': email['from'],
                'category': category,
                'subcategory': subcategory,
                'confidence': f"{confidence:.2%}",
                'is_duplicate': is_duplicate
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def provide_feedback():
    try:
        data = request.json
        if not data or not all(k in data for k in ['text', 'category', 'subcategory']):
            return jsonify({"error": "Missing required fields"}), 400
        text = data['text']
        category = data['category']
        subcategory = data['subcategory']
        processor.update_with_feedback(text, category, subcategory)
        return jsonify({"message": "Feedback processed successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_folder', methods=['POST'])
def process_folder():
    folder_path = request.form.get('folder_path')
    if not folder_path or not os.path.isdir(folder_path):
        return render_template('results.html', error="Invalid or missing folder path", results=[], categories=processor.categories)
    
    results = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.msg'):
                file_path = os.path.join(folder_path, filename)
                msg = extract_msg.Message(file_path)
                subject = msg.subject or 'No Subject'
                from_addr = msg.sender or 'Unknown'
                body = msg.body.strip() if msg.body else 'No Content'
                attachments = [att.longFilename or att.shortFilename for att in msg.attachments] if msg.attachments else []
                
                email = {'subject': subject, 'from': from_addr, 'content': body}
                category, subcategory, confidence = processor.categorize_email(email)
                is_duplicate = processor.detect_duplicate(email)
                results.append({
                    'filename': filename,
                    'subject': subject,
                    'from': from_addr,
                    'text': f"{subject} {body}",  # For feedback
                    'category': category,
                    'subcategory': subcategory,
                    'confidence': f"{confidence:.2%}",
                    'is_duplicate': is_duplicate,
                    'attachments': attachments
                })
                msg.close()  # Close the message to free resources
        return render_template('results.html', results=results, error=None, categories=processor.categories)
    except Exception as e:
        return render_template('results.html', error=str(e), results=[], categories=processor.categories)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        text = request.form.get('text')
        category = request.form.get('category')
        subcategory = request.form.get('subcategory')
        if not all([text, category, subcategory]):
            return render_template('results.html', error="Missing feedback fields", results=[], categories=processor.categories)
        if category not in processor.categories or subcategory not in processor.categories[category]:
            return render_template('results.html', error=f"Invalid pair: {category}_{subcategory}", results=[], categories=processor.categories)
        processor.update_with_feedback(text, category, subcategory)
        return redirect(url_for('index'))
    except Exception as e:
        return render_template('results.html', error=f"Feedback failed: {str(e)}", results=[], categories=processor.categories)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)