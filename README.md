# Legal AI Fact-Checker with Explainable AI (XAI)

## 🚀 Overview

A comprehensive legal document analysis system that combines PDF processing, AI-powered claim extraction, fact verification, and explainable AI (XAI) to provide transparent insights into legal document analysis. The system features advanced counterfactual explanations, bias analysis, and interactive visualizations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEGAL AI XAI SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: User Interface (Flask Web App)                       │
│  ├── Upload Interface (Drag & Drop PDF)                        │
│  ├── XAI Dashboard (Interactive Charts)                        │
│  └── Results Display (Counterfactual Explanations)             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Web Service Layer                                    │
│  ├── simple_web_interface.py (Flask App - Port 5003)          │
│  ├── Request Handling & Response Normalization                 │
│  └── Template Rendering & Visualization                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: ML Service Layer (FastAPI - Port 8000)              │
│  ├── PDF Processing Endpoints                                  │
│  ├── Comprehensive XAI Analysis                                │
│  ├── Counterfactual Generation                                 │
│  └── Bias & Uncertainty Analysis                               │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: AI Model Layer                                       │
│  ├── Legal-BERT (Claim Extraction)                            │
│  ├── RoBERTa-MNLI (Fact Verification)                         │
│  ├── InLegalBERT (Legal Embeddings)                           │
│  └── Sentence Transformers (Similarity)                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: XAI Engine                                          │
│  ├── SHAP (Feature Attribution)                               │
│  ├── LIME (Local Explanations)                                │
│  ├── Attention Visualization                                   │
│  ├── Counterfactual Generator (Enhanced)                      │
│  ├── Uncertainty Analysis                                      │
│  └── Bias Detection & Mitigation                              │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Data Layer                                          │
│  ├── SQLite Database (Legal Cases)                            │
│  ├── FAISS Vector Index (Semantic Search)                     │
│  ├── PDF Storage & Processing                                  │
│  └── RAG (Retrieval-Augmented Generation)                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Key Features

### 1. **Advanced PDF Processing**
- Multi-format PDF support with robust error handling
- Intelligent clause extraction using Legal-BERT
- Entity recognition for legal terms and concepts
- Page-level analysis and structure preservation

### 2. **Explainable AI (XAI) Components**
- **SHAP Values**: Global and local feature importance
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Attention Visualization**: Transformer attention weight analysis
- **Decision Trees**: Human-readable decision paths
- **Feature Importance**: TF-IDF and model-based feature ranking

### 3. **Enhanced Counterfactual Explanations** 🆕 **RECENTLY IMPLEMENTED**
- **Word Substitution**: Legal term alternatives and impacts
  ```
  Original: "The party shall pay damages..."
  Modified: "The party may pay damages..."
  Impact: HIGH - Changes legal obligation strength
  ```
- **Contextual Scenarios**: Alternative contract conditions
  ```
  "What if liability was limited instead of unlimited?"
  "What if the payment term was 60 days instead of 30?"
  ```
- **Negation Analysis**: Opposite legal interpretations
  ```
  "liable" → "not liable"
  "required" → "not required"
  ```
- **Targeted Outcomes**: Specific result-oriented modifications
  - Generate changes that would **support** the claim
  - Generate changes that would **refute** the claim
- **Impact Assessment**: High/Medium/Low change categorization
- **Enhanced Visualization**: Color-coded counterfactual cards with explanations

### 4. **Current XAI Dashboard Features** 🎯
- **Interactive Tabs**: Reasoning, Features, **Counterfactuals**, LIME, Bias
- **Professional Charts**: Chart.js with legal color themes
- **Real-time Updates**: Dynamic content loading
- **Mobile Responsive**: Works on tablets and phones

### 4. **Bias Analysis & Fairness**
- Demographic neutrality scoring
- Feature balance assessment
- Representation fairness metrics
- Bias mitigation suggestions
- Fair outcome evaluation

### 5. **Uncertainty Quantification**
- Confidence interval estimation
- Prediction stability analysis
- Monte Carlo uncertainty sampling
- Entropy-based uncertainty measures

### 6. **Interactive Visualizations**
- Professional Chart.js visualizations
- Color-coded importance distributions
- Confidence trend analysis
- Page-level heatmaps
- Real-time chart animations

## 📁 Current Project Structure (After Cleanup)

```
legal-factcheck-mcp/
├── 📄 README.md                          # This comprehensive documentation
├── 📄 requirements.txt                   # Python dependencies
├── 🐍 simple_web_interface.py           # 🎯 MAIN Flask web application (Port 5003)
│
├── 📁 backup_complex/src/                # 🎯 CORE ML Services
│   ├── 🤖 ml_service.py                  # 🎯 FastAPI ML service (Port 8000) 
│   ├── 📄 pdf_rag_processor.py           # PDF processing & RAG system
│   ├── 🧠 xai_explainer.py              # XAI engine with counterfactuals
│   └── 📊 document_tracker.py            # Document management utilities
│
├── 📁 templates/                         # 🎯 Web Templates (Flask)
│   ├── 🎨 xai_dashboard.html            # Upload interface with drag & drop
│   ├── 📊 xai_results.html              # Results with counterfactual explanations
│   └── 🎨 base.html                     # Base template layout
│
├── 📁 uploads/                          # PDF file storage directory
├── 📁 .venv/                           # Python virtual environment
│
├── 🗃️ legal_database.db                # SQLite legal cases database
├── 🗃️ document_analysis.db             # Analysis results storage
│
├── 📄 FINAL_SOLUTION.md                 # Additional documentation
├── 📄 XAI_DASHBOARD_SOLUTION.md         # XAI implementation notes
│
└── 📄 Sample Documents/                  # Test PDFs
    ├── sample_contract.pdf              # Basic contract testing
    ├── complex_legal_contract.pdf       # Complex multi-clause docs
    ├── pages-29-deed-sample.pdf         # Property law sample
    └── Legal notice - Brigade...pdf     # Real-world legal notice
```

## 🎯 **Currently Active Files** (What Actually Runs)

### **Primary Application Files:**
1. **`simple_web_interface.py`** - Flask web server (Port 5003)
2. **`backup_complex/src/ml_service.py`** - FastAPI ML service (Port 8000)  
3. **`backup_complex/src/pdf_rag_processor.py`** - PDF processing engine
4. **`backup_complex/src/xai_explainer.py`** - Enhanced XAI with counterfactuals

### **Web Interface Templates:**
1. **`templates/xai_dashboard.html`** - Main upload interface
2. **`templates/xai_results.html`** - Results display with counterfactuals
3. **`templates/base.html`** - Shared template layout

## 🛠️ Technical Dependencies

### **Current requirements.txt:**
```bash
# Core Web Frameworks
fastapi==0.104.1                    # ML Service API framework
uvicorn[standard]==0.24.0           # ASGI server for FastAPI
flask==3.0.0                       # Web interface framework
werkzeug==3.0.1                    # Flask utilities

# AI/ML Models & Processing
transformers==4.35.0               # Hugging Face transformers (Legal-BERT, RoBERTa)
sentence-transformers==2.2.2       # Sentence embeddings (InLegalBERT)
torch==2.1.0                      # PyTorch backend
faiss-cpu==1.7.4                  # Vector similarity search
scikit-learn==1.3.2               # ML utilities

# Explainable AI (XAI)
shap==0.43.0                       # SHAP explanations
lime==0.2.0.1                      # LIME local explanations

# PDF Processing
PyPDF2==3.0.1                     # PDF text extraction
pymupdf==1.23.0                   # Advanced PDF processing  
pdfplumber==0.10.3                # PDF structure analysis

# Data & Visualization
numpy==1.25.0                     # Numerical computing
matplotlib==3.8.0                 # Plotting backend
seaborn==0.13.0                   # Statistical visualization
plotly==5.17.0                    # Interactive charts

# Language Processing
langchain==0.0.350                # LLM integration framework
langchain-community==0.0.1        # Community LangChain tools
tiktoken==0.5.1                   # Token counting
textstat==0.7.3                   # Text statistics

# Database & Storage
chromadb==0.4.18                  # Vector database
pydantic==2.5.0                   # Data validation

# Utilities
requests==2.31.0                  # HTTP client
python-multipart==0.0.6           # File upload handling
dash==2.14.2                      # Dashboard components
```

## 🎯 System Requirements
- **Python**: 3.8+ (tested on 3.11, 3.12)
- **RAM**: 4GB minimum, 8GB recommended (for AI models)
- **Storage**: 2GB for models, 1GB for dependencies
- **Network**: Internet required for initial model downloads
- **OS**: macOS (tested), Linux, Windows supported

### Quick Start (Current System)

1. **Navigate to Project Directory**
```bash
cd "/Users/a91788/Desktop/Final Year /XAI/legal-factcheck-mcp"
```

2. **Activate Virtual Environment**
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

3. **Install Dependencies** (if not already installed)
```bash
pip install -r requirements.txt
```

4. **Start ML Service** (Terminal 1) 🎯
```bash
# Ensure you're in the project root directory
cd "/Users/a91788/Desktop/Final Year /XAI/legal-factcheck-mcp"
source .venv/bin/activate
python backup_complex/src/ml_service.py
```
*⏳ Wait for models to load (2-3 minutes on first run)*
*✅ Look for "INFO:uvicorn.server:Uvicorn running on http://0.0.0.0:8000"*

5. **Start Web Interface** (Terminal 2) 🎯
```bash
# Open new terminal, navigate to project root
cd "/Users/a91788/Desktop/Final Year /XAI/legal-factcheck-mcp"
source .venv/bin/activate
python simple_web_interface.py
```
*✅ Look for "Running on http://127.0.0.1:5003"*

6. **Access the System** 🚀
```
📊 Main Dashboard: http://localhost:5003
🔧 ML Service API:  http://localhost:8000/docs (FastAPI docs)
```

## ⚡ **Current System Status Check**

### **Verify Services Are Running:**
```bash
# Check ML service (should return JSON health status)
curl http://localhost:8000/health

# Check web interface (should return HTML)
curl http://localhost:5003/

# Check if both ports are active
lsof -i :8000  # ML Service
lsof -i :5003  # Web Interface
```

### **Expected Output:**
- **ML Service**: Models loading messages, then "Uvicorn running on http://0.0.0.0:8000"
- **Web Interface**: "🚀 Starting Legal AI XAI Dashboard..." then Flask server startup

## 🔧 Current API Endpoints (Live System)

### **ML Service (Port 8000) - backup_complex/src/ml_service.py**

| Endpoint | Method | Status | Description |
|----------|--------|--------|-------------|
| `/process-pdf` | POST | ✅ Active | Extract and analyze legal clauses from PDF |
| `/comprehensive-xai-explanation` | POST | ✅ Active | Generate complete XAI explanations |
| `/generate-counterfactuals` | POST | 🆕 **Enhanced** | **Advanced counterfactual examples with categorization** |
| `/analyze-bias` | POST | ✅ Active | Bias detection and fairness analysis |
| `/uncertainty-analysis` | POST | ✅ Active | Uncertainty quantification and confidence intervals |
| `/health` | GET | ✅ Active | Service health check |
| `/docs` | GET | ✅ Active | FastAPI interactive documentation |

### **Web Interface (Port 5003) - simple_web_interface.py**

| Route | Method | Status | Description |
|-------|--------|--------|-------------|
| `/` | GET | ✅ Active | Main upload dashboard with drag & drop |
| `/upload` | POST | ✅ Enhanced | PDF upload with **counterfactual integration** |
| `/api/bias-analysis` | POST | ✅ Active | Bias analysis API endpoint |

### **🆕 Enhanced Counterfactual Endpoint Response:**
```json
{
  "original_claim": "The party shall pay damages...",
  "counterfactual_examples": [
    {
      "text": "The party may pay damages...",
      "type": "word_substitution", 
      "changes": ["'shall' → 'may'"],
      "impact": "high",
      "explanation": "Changes legal obligation strength",
      "likelihood": "likely"
    }
  ],
  "categorized_counterfactuals": {
    "word_substitution": [...],
    "contextual_scenario": [...], 
    "negation": [...],
    "targeted_support": [...],
    "targeted_refute": [...]
  },
  "impact_statistics": {
    "high_impact": 3,
    "medium_impact": 2, 
    "low_impact": 1,
    "total_examples": 6
  },
  "status": "success"
}
```

## 🧠 XAI Explanation Types

### 1. **Decision Summary**
- Human-readable analysis conclusion
- Confidence score with visual meter
- Key reasoning steps enumeration

### 2. **Feature Importance**
- TF-IDF based term significance
- Legal keyword importance scoring
- Interactive feature bars with gradients

### 3. **Counterfactual Explanations** 🌟
```python
# Example counterfactual types:
{
  "word_substitution": [
    {
      "text": "The party may pay damages...",  # Changed "shall" to "may"
      "changes": ["'shall' → 'may'"],
      "impact": "high",
      "explanation": "Changes legal obligation strength"
    }
  ],
  "contextual_scenario": [
    {
      "text": "What if liability was limited instead of unlimited?",
      "impact": "medium",
      "likelihood": "possible"
    }
  ],
  "targeted_support": [...],  # Modifications supporting the claim
  "targeted_refute": [...]    # Modifications refuting the claim
}
```

### 4. **LIME Explanations**
- Word-level importance visualization
- Color-coded text highlighting
- Local model approximations

### 5. **Bias Analysis**
- Demographic neutrality scoring (0-1)
- Fairness metrics across categories
- Bias mitigation recommendations

### 6. **Uncertainty Analysis**
- Prediction stability assessment
- 95% confidence intervals
- Entropy-based uncertainty measures

## 📊 Visualization Dashboard

### Interactive Charts (Chart.js)
1. **Clause Importance Distribution**
   - Color-coded by criticality (Red/Yellow/Blue/Green)
   - Animated bar charts with gradients
   - Real-time filtering capabilities

2. **Clause Types Distribution**
   - Professional pie chart with hover effects
   - Legal category breakdown
   - Dynamic legend with click interactions

3. **Confidence Score Trends**
   - Line chart with trend analysis
   - Confidence intervals visualization
   - Smooth animations and tooltips

4. **Page Distribution Analysis**
   - Bubble chart for page-level insights
   - Size indicates clause density
   - Interactive hover information

## 🤖 AI Models Used

### Core Models
- **Legal-BERT**: `nlpaueb/legal-bert-base-uncased`
- **RoBERTa-MNLI**: `roberta-large-mnli`
- **InLegalBERT**: `law-ai/InLegalBERT`
- **Sentence Transformers**: Various legal domain models

### Model Pipeline
1. **Document Processing**: PyMuPDF → Text extraction
2. **Clause Extraction**: Legal-BERT → Entity recognition
3. **Fact Verification**: RoBERTa-MNLI → Claim validation
4. **Embedding Generation**: InLegalBERT → Semantic vectors
5. **Similarity Search**: FAISS → Relevant case retrieval
6. **XAI Generation**: Multiple techniques → Explanations

## 🔍 Counterfactual Generation Process

### Algorithm Overview
```python
def generate_counterfactuals(original_text, num_examples=5):
    counterfactuals = []
    
    # 1. Word Substitution Analysis
    for legal_term in LEGAL_MODIFIERS:
        if legal_term in original_text:
            alternatives = get_alternatives(legal_term)
            modified_text = substitute_term(original_text, legal_term, alternatives)
            impact = assess_legal_impact(legal_term, alternatives)
            counterfactuals.append({
                'text': modified_text,
                'type': 'word_substitution',
                'impact': impact,
                'changes': [f"'{legal_term}' → '{alternative}'"]
            })
    
    # 2. Contextual Scenario Generation
    context = detect_legal_context(original_text)
    scenarios = generate_scenarios(context)
    counterfactuals.extend(scenarios)
    
    # 3. Negation Analysis
    negated_versions = generate_negations(original_text)
    counterfactuals.extend(negated_versions)
    
    return categorize_and_rank(counterfactuals)
```

### Legal Term Modifications
- **Obligation Strength**: shall ↔ may ↔ should ↔ must
- **Liability Terms**: liable ↔ responsible ↔ not liable
- **Temporal**: immediately ↔ within 30 days ↔ eventually
- **Scope**: all ↔ some ↔ specific ↔ any

## 🛡️ Error Handling & Robustness

### Graceful Degradation
- Missing model fallbacks
- Network timeout handling
- Malformed PDF recovery
- Empty result normalization

### Response Normalization
```python
def normalize_xai_response(xai_data):
    return {
        'decision_summary': xai_data.get('decision_summary', 'Analysis completed'),
        'confidence_score': max(0.0, min(1.0, xai_data.get('confidence_score', 0.5))),
        'counterfactual_examples': xai_data.get('counterfactual_examples', []),
        'feature_importance': xai_data.get('feature_importance', {}),
        # ... robust defaults for all fields
    }
```

## 📈 Performance Optimizations

### Model Loading
- Lazy loading of heavy models
- Model caching and reuse
- GPU acceleration (MPS/CUDA)
- Memory-efficient processing

### Web Performance
- Async request handling
- Response compression
- Chart.js optimization
- Template caching

## 🔒 Security Considerations

- File type validation (PDF only)
- File size limits (50MB)
- Secure filename handling
- Request timeout enforcement
- Input sanitization

## 🧪 Testing & Validation

### Test Documents
- `sample_contract.pdf`: Basic contract analysis
- `complex_legal_contract.pdf`: Multi-clause documents
- `pages-29-deed-sample.pdf`: Property law testing

### Validation Methods
- Counterfactual accuracy testing
- XAI explanation consistency
- Bias detection validation
- Performance benchmarking

## 🚨 Troubleshooting (Current System)

### **Common Issues & Solutions:**

#### 1. **ML Service Won't Start (Port 8000)**
```bash
# Check if port is already in use
lsof -i :8000

# Kill existing process if needed
lsof -ti :8000 | xargs kill -9

# Restart with verbose output
cd "/Users/a91788/Desktop/Final Year /XAI/legal-factcheck-mcp"
source .venv/bin/activate
python backup_complex/src/ml_service.py
```

#### 2. **Web Interface Won't Start (Port 5003)**
```bash
# Check if port is already in use
lsof -i :5003

# Kill existing process if needed  
lsof -ti :5003 | xargs kill -9

# Restart Flask app
python simple_web_interface.py
```

#### 3. **Models Not Loading (First Time Setup)**
- **Issue**: Long loading time or download failures
- **Solution**: Ensure stable internet, wait 5-10 minutes
- **Check**: Monitor terminal output for download progress

#### 4. **Counterfactuals Not Showing**
- **Issue**: XAI tab shows no counterfactual examples
- **Solution**: 
  ```bash
  # Test counterfactual endpoint directly
  curl -X POST http://localhost:8000/generate-counterfactuals \
    -H "Content-Type: application/json" \
    -d '{"claim": "The party shall pay damages for breach."}'
  ```
- **Expected**: JSON response with counterfactual examples

#### 5. **PDF Upload Fails**
- **Check**: File size < 50MB, valid PDF format
- **Debug**: Check browser console and Flask terminal output
- **Test**: Try with provided sample PDFs first

### **Debug Commands (Current System):**
```bash
# Quick health check
curl http://localhost:8000/health
curl http://localhost:5003/

# Check running processes  
ps aux | grep "ml_service\|simple_web_interface"

# Check virtual environment
which python  # Should show .venv path
pip list | grep -E "fastapi|flask|transformers"

# Test sample document processing
curl -X POST http://localhost:8000/process-pdf \
  -F "file=@sample_contract.pdf"
```

### **Expected Terminal Outputs:**

#### **ML Service Startup (backup_complex/src/ml_service.py):**
```
INFO:__main__:Loaded claim extraction model: nlpaueb/legal-bert-base-uncased
INFO:sentence_transformers.SentenceTransformer:Use pytorch device_name: mps
INFO:__main__:Loaded legal sentence transformer
INFO:__main__:Built FAISS index with X cases
INFO:__main__:Loaded explanation generation model
INFO:uvicorn.server:Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### **Web Interface Startup (simple_web_interface.py):**
```
🚀 Starting Legal AI XAI Dashboard...
📊 Dashboard URL: http://localhost:5003
🔧 ML Service URL: http://localhost:8000
📁 Upload folder: uploads
✅ ML service is running
* Running on http://127.0.0.1:5003
```

## 🌟 Recent System Enhancements (September 2025)

### **🆕 Version 2.0 Features:**

#### **1. Enhanced Counterfactual System**
- **Legal-Domain Specific**: Understands legal term relationships
- **Multi-Type Generation**: 
  - Word substitution (`shall` → `may`, `liable` → `not liable`)
  - Contextual scenarios (liability caps, payment terms)
  - Negation analysis (opposite interpretations)
  - Targeted outcomes (support/refute specific claims)
- **Impact Assessment**: Automatic categorization (High/Medium/Low)
- **Professional UI**: Color-coded cards with detailed explanations

#### **2. System Architecture Improvements**
- **Cleaned Codebase**: Removed 20+ redundant files
- **Streamlined Structure**: Clear separation of concerns
- **Enhanced Error Handling**: Graceful degradation and normalization
- **Performance Optimization**: Async request handling

#### **3. Documentation & Usability**
- **Comprehensive README**: This document with exact file paths
- **Real-World Testing**: Verified with actual legal documents
- **Interactive Visualization**: Chart.js integration with legal themes
- **Mobile Responsiveness**: Works across all device types

### **🔄 System Workflow (Current)**
```
1. User uploads PDF → templates/xai_dashboard.html
2. Flask processes → simple_web_interface.py
3. ML analysis → backup_complex/src/ml_service.py
4. PDF processing → backup_complex/src/pdf_rag_processor.py  
5. XAI generation → backup_complex/src/xai_explainer.py
6. Counterfactuals → Enhanced counterfactual generator
7. Results display → templates/xai_results.html (with counterfactuals)
8. Interactive charts → Chart.js visualizations
```

### **💡 Key Innovations Implemented:**
- **Legal Context Awareness**: System understands contract vs liability contexts
- **Probabilistic Explanations**: Likelihood assessments for counterfactuals
- **Multi-Modal XAI**: SHAP + LIME + Counterfactuals + Bias analysis
- **Production-Ready Error Handling**: Handles malformed PDFs, network issues
- **Scalable Architecture**: FastAPI backend with Flask frontend

## 🔮 Future Enhancements

- **Multi-language Support**: International legal systems
- **Advanced RAG**: GPT integration for enhanced explanations
- **Real-time Collaboration**: Multi-user document analysis
- **API Gateway**: Enterprise integration capabilities
- **Blockchain Integration**: Immutable analysis records

## 📞 Support & Documentation

- **Technical Issues**: Check troubleshooting section
- **Model Updates**: Monitor Hugging Face repositories
- **Feature Requests**: Submit GitHub issues
- **Performance**: Monitor system resources during analysis

## 📜 License & Usage

This project is designed for educational and research purposes in legal AI and explainable machine learning. Ensure compliance with local regulations when processing legal documents.

---

**Built with ❤️ for Transparent Legal AI**

*Last Updated: September 2025*
*Version: 2.0.0 (Enhanced Counterfactual Release)*

## System Architecture

### Core Components

1. **Web Interface** (`app.py`)
   - Flask-based web application
   - File upload and processing interface
   - Results visualization

2. **ML Service** (`ml_service.py`)
   - FastAPI-based REST API
   - Legal document processing
   - Claim extraction and verification
   - XAI explanations

3. **Supporting Modules**
   - `pdf_rag_processor.py`: PDF processing with RAG
   - `xai_explainer.py`: Explainable AI components
   - `setup_database.py`: Legal case database setup

### Features

- **PDF Document Analysis**: Upload and analyze legal PDFs
- **Clause Extraction**: Automatic identification of important legal clauses
- **Fact Verification**: AI-powered claim verification against legal precedents
- **XAI Explanations**: Comprehensive explainable AI features including:
  - SHAP values
  - LIME explanations
  - Attention weights
  - Counterfactual examples
  - Uncertainty analysis
  - Bias detection

## Database

The system uses SQLite with sample legal cases:
- Hadley v. Baxendale (Contract Law)
- Miranda v. Arizona (Criminal Procedure)
- Brown v. Board of Education (Constitutional Law)
- Palsgraf v. Long Island Railroad Co. (Tort Law)
- And more...

## API Endpoints

### ML Service (Port 8000)
- `POST /process-pdf`: Process PDF documents
- `POST /verify-claim`: Verify legal claims
- `POST /extract-claims`: Extract claims from text
- `POST /comprehensive-xai-explanation`: Generate XAI explanations
- `GET /health`: Health check

### Web Interface (Port 5001)
- `GET /`: Main upload page
- `POST /upload`: Handle PDF uploads
- `POST /analyze_clause`: Analyze specific clauses
- `GET /health`: Health check

## File Structure (Cleaned)
```
legal-factcheck-mcp/
├── app.py                    # Main web application
├── ml_service.py             # ML API service
├── pdf_rag_processor.py      # PDF processing
├── xai_explainer.py          # XAI components
├── setup_database.py         # Database initialization
├── start_system.py           # Startup script
├── requirements.txt          # Python dependencies
├── legal_database.db         # Legal case database
├── templates/                # Web templates
│   ├── base.html
│   ├── index.html
│   └── results.html
├── uploads/                  # PDF upload directory
└── web_interface/            # Alternative web interface
    └── templates/
```

## Cleaned Up Files Removed
- `simple_app.py` (empty)
- `simple_app_xai.py` (empty)
- `simple_requirements.txt` (empty)
- `enhanced_pdf_test.py`
- `test_pdf_extraction.py`
- `robust_pdf_processor.py`
- `templates/` (duplicate folder)

## Usage

1. **Upload PDF**: Go to http://localhost:5001 and upload a legal PDF
2. **Process Document**: The system will extract clauses and analyze importance
3. **View Results**: See extracted clauses with importance scores and explanations
4. **Analyze Clauses**: Click on individual clauses for detailed XAI analysis

## Models Used

- **Legal-BERT**: For legal text understanding
- **Sentence Transformers**: For semantic similarity
- **RoBERTa-MNLI**: For textual entailment
- **Custom XAI Models**: For explainability features

## System Status
✅ ML Service: Running on port 8000
✅ Web Interface: Running on port 5001  
✅ Database: Initialized with sample cases
✅ All AI models: Loaded and ready

## Troubleshooting

### Port Conflicts
```bash
# Kill processes on specific ports
lsof -ti:8000 | xargs kill -9  # ML Service
lsof -ti:5001 | xargs kill -9  # Web Interface
```

### Model Loading Issues
- Models will download automatically on first run
- Ensure stable internet connection for initial setup
- Check console output for loading progress

### Dependencies
```bash
# Reinstall requirements if needed
pip install -r requirements.txt --force-reinstall
```

## Development Notes

- The system is configured for development with debug mode enabled
- For production deployment, disable debug mode and use proper WSGI server
- Add authentication and authorization for production use
- Consider rate limiting for API endpoints
