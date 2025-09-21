#!/usr/bin/env python3
"""
Simple Legal AI Web Interface with XAI Dashboard
Combines PDF upload with comprehensive explainable AI analysis
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import requests
import json
import os
from datetime import datetime
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'legal-ai-xai-dashboard-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configuration
ML_SERVICE_URL = "http://localhost:8000"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_ml_service():
    """Check if ML service is running"""
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def normalize_pdf_results(results):
    """Normalize PDF processing results to handle missing attributes"""
    normalized = {
        'filename': results.get('filename', 'Unknown'),
        'document_info': {
            'total_pages': results.get('document_info', {}).get('total_pages', 0),
            'total_clauses_found': results.get('document_info', {}).get('total_clauses_found', 
                                              len(results.get('important_clauses', []))),
            'processing_time': results.get('document_info', {}).get('processing_time', 0.0),
        },
        'clause_summary': {
            'total_clauses': results.get('clause_summary', {}).get('total_clauses', 
                           len(results.get('important_clauses', []))),
            'avg_importance_score': results.get('clause_summary', {}).get('avg_importance_score', 
                                  calculate_avg_importance(results.get('important_clauses', []))),
            'clause_types_distribution': results.get('clause_summary', {}).get('clause_types_distribution', 
                                       get_clause_types_distribution(results.get('important_clauses', []))),
            'pages_with_clauses': results.get('clause_summary', {}).get('pages_with_clauses', 
                                 len(set([c.get('page_number', 1) for c in results.get('important_clauses', [])])))
        },
        'important_clauses': normalize_clauses(results.get('important_clauses', [])),
        'status': results.get('status', 'success')
    }
    return normalized

def normalize_clauses(clauses):
    """Normalize clause data to handle missing attributes"""
    normalized_clauses = []
    for clause in clauses:
        normalized_clause = {
            'text': clause.get('text', 'No text available'),
            'importance_score': clause.get('importance_score', 0.5),
            'type': clause.get('type', 'general'),
            'page_number': clause.get('page_number', 1),
            'entities': clause.get('entities', []),
            'confidence': clause.get('confidence', 0.5)
        }
        normalized_clauses.append(normalized_clause)
    return normalized_clauses

def normalize_xai_response(xai_data):
    """Normalize XAI response to handle missing attributes"""
    return {
        'decision_summary': xai_data.get('decision_summary', 'No summary available'),
        'confidence_score': xai_data.get('confidence_score', 0.5),
        'reasoning_steps': xai_data.get('reasoning_steps', ['Analysis performed']),
        'feature_importance': xai_data.get('feature_importance', {}),
        'attention_weights': xai_data.get('attention_weights', {}),
        'counterfactual_examples': xai_data.get('counterfactual_examples', []),
        'decision_tree_path': xai_data.get('decision_tree_path', 'Standard analysis path'),
        'lime_explanation': {
            'word_importance': xai_data.get('lime_explanation', {}).get('word_importance', {})
        },
        'shap_values': xai_data.get('shap_values', {}),
        'uncertainty_analysis': xai_data.get('uncertainty_analysis', {
            'prediction_stability': 'Medium',
            'confidence_interval_95': [0.4, 0.6],
            'entropy': 0.5
        }),
        'bias_analysis': {
            'demographic_neutrality_score': xai_data.get('bias_analysis', {}).get('demographic_neutrality_score', 0.8),
            'fairness_metrics': {
                'feature_balance': xai_data.get('bias_analysis', {}).get('fairness_metrics', {}).get('feature_balance', 0.8),
                'representation_fairness': xai_data.get('bias_analysis', {}).get('fairness_metrics', {}).get('representation_fairness', 0.8),
                'outcome_equity': xai_data.get('bias_analysis', {}).get('fairness_metrics', {}).get('outcome_equity', 0.8)
            }
        },
        'legal_precedent_chain': xai_data.get('legal_precedent_chain', []),
        'alternative_interpretations': xai_data.get('alternative_interpretations', [])
    }

def calculate_avg_importance(clauses):
    """Calculate average importance score from clauses"""
    if not clauses:
        return 0.0
    scores = [clause.get('importance_score', 0.5) for clause in clauses]
    return sum(scores) / len(scores)

def get_clause_types_distribution(clauses):
    """Get distribution of clause types"""
    if not clauses:
        return {}
    
    type_counts = {}
    for clause in clauses:
        clause_type = clause.get('type', 'general')
        type_counts[clause_type] = type_counts.get(clause_type, 0) + 1
    
    return type_counts

@app.route('/')
def index():
    """Main upload page"""
    ml_status = check_ml_service()
    return render_template('xai_dashboard.html', ml_service_status=ml_status)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and processing"""
    try:
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Get processing parameters
            min_importance = float(request.form.get('min_importance', 0.5))
            jurisdiction = request.form.get('jurisdiction', 'federal')
            
            # Process PDF
            with open(filepath, 'rb') as pdf_file:
                files = {'file': (filename, pdf_file, 'application/pdf')}
                data = {'min_importance': min_importance}
                
                response = requests.post(
                    f"{ML_SERVICE_URL}/process-pdf",
                    files=files,
                    data=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    pdf_results = response.json()
                    
                    # Normalize the response structure to handle missing attributes
                    normalized_results = normalize_pdf_results(pdf_results)
                    
                    # Get XAI dashboard data
                    try:
                        dashboard_response = requests.get(f"{ML_SERVICE_URL}/xai-dashboard-data", timeout=10)
                        dashboard_data = dashboard_response.json() if dashboard_response.status_code == 200 else {}
                    except:
                        dashboard_data = {}
                    
                    # Generate comprehensive XAI explanations for top clauses
                    xai_explanations = []
                    important_clauses = normalized_results.get('important_clauses', [])
                    
                    for clause in important_clauses[:3]:  # Top 3 clauses
                        try:
                            # Get comprehensive XAI explanation
                            xai_response = requests.post(
                                f"{ML_SERVICE_URL}/comprehensive-xai-explanation",
                                json={
                                    'claim': clause.get('text', ''),
                                    'jurisdiction': jurisdiction
                                },
                                timeout=30
                            )
                            
                            if xai_response.status_code == 200:
                                xai_data = xai_response.json()
                                normalized_xai = normalize_xai_response(xai_data.get('xai_explanation', {}))
                                
                                # Get enhanced counterfactual explanations
                                try:
                                    counterfactual_response = requests.post(
                                        f"{ML_SERVICE_URL}/generate-counterfactuals",
                                        json={
                                            'claim': clause.get('text', ''),
                                            'jurisdiction': jurisdiction
                                        },
                                        timeout=15
                                    )
                                    
                                    if counterfactual_response.status_code == 200:
                                        counterfactual_data = counterfactual_response.json()
                                        # Merge enhanced counterfactuals into XAI response
                                        normalized_xai['counterfactual_examples'] = counterfactual_data.get('counterfactual_examples', [])
                                        normalized_xai['counterfactual_categories'] = counterfactual_data.get('categorized_counterfactuals', {})
                                        normalized_xai['counterfactual_impact_stats'] = counterfactual_data.get('impact_statistics', {})
                                except Exception as cf_error:
                                    print(f"Counterfactual fetch error: {cf_error}")
                                
                                xai_explanations.append({
                                    'clause': clause,
                                    'xai': normalized_xai
                                })
                        except Exception as e:
                            print(f"XAI explanation error for clause: {e}")
                            continue
                    
                    return render_template('xai_results.html', 
                                         filename=filename,
                                         pdf_results=normalized_results,
                                         xai_explanations=xai_explanations,
                                         dashboard_data=dashboard_data,
                                         min_importance=min_importance,
                                         jurisdiction=jurisdiction)
                else:
                    flash(f'Processing failed: {response.text}')
                    return redirect(url_for('index'))
        else:
            flash('Invalid file type. Please upload a PDF file.')
            return redirect(url_for('index'))
            
    except Exception as e:
        error_msg = f'Upload error: {str(e)}'
        print(f"Detailed error: {traceback.format_exc()}")
        flash(error_msg)
        return redirect(url_for('index'))

@app.route('/api/bias-analysis', methods=['POST'])
def bias_analysis():
    """Get bias analysis for a specific text"""
    try:
        data = request.get_json()
        response = requests.post(f"{ML_SERVICE_URL}/analyze-bias", json=data, timeout=30)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/uncertainty-analysis', methods=['POST'])
def uncertainty_analysis():
    """Get uncertainty analysis for a specific claim"""
    try:
        data = request.get_json()
        response = requests.post(f"{ML_SERVICE_URL}/uncertainty-analysis", json=data, timeout=30)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-importance', methods=['POST'])
def feature_importance():
    """Get feature importance analysis"""
    try:
        data = request.get_json()
        response = requests.post(f"{ML_SERVICE_URL}/feature-importance", json=data, timeout=30)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard-data')
def dashboard_data():
    """Get current dashboard data"""
    try:
        response = requests.get(f"{ML_SERVICE_URL}/xai-dashboard-data", timeout=10)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Legal AI XAI Dashboard...")
    print("📊 Dashboard URL: http://localhost:5003")
    print("🔧 ML Service URL:", ML_SERVICE_URL)
    print("📁 Upload folder:", UPLOAD_FOLDER)
    
    if check_ml_service():
        print("✅ ML service is running")
    else:
        print("❌ ML service is not responding - please start it first")
    
    app.run(host='0.0.0.0', port=5003, debug=True)
