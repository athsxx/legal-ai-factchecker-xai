# pdf_rag_processor.py - PDF processing with RAG for legal clause extraction

import PyPDF2
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
import re
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ExtractedClause:
    text: str
    clause_type: str
    importance_score: float
    page_number: int
    section: str
    entities: List[Dict]
    context: str

class PDFProcessor:
    """Extract text and structure from PDF legal documents"""
    
    def __init__(self):
        self.legal_section_patterns = {
            'definitions': r'(?i)(definitions?|defined terms?)',
            'obligations': r'(?i)(obligations?|duties|shall|must|required to)',
            'rights': r'(?i)(rights?|entitled to|may)',
            'termination': r'(?i)(termination|terminate|end|expir)',
            'payment': r'(?i)(payment|pay|fee|cost|amount|compensation)',
            'liability': r'(?i)(liability|liable|responsible|damages)',
            'dispute': r'(?i)(dispute|arbitration|litigation|court)',
            'governing_law': r'(?i)(governing law|jurisdiction|applicable law)',
            'force_majeure': r'(?i)(force majeure|act of god|unforeseeable)',
            'confidentiality': r'(?i)(confidential|non-disclosure|proprietary)'
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Extract text and metadata from PDF"""
        try:
            # Try PyMuPDF first (better for complex layouts)
            doc = fitz.open(pdf_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Extract additional metadata
                blocks = page.get_text("dict")
                
                pages_content.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'word_count': len(text.split()),
                    'blocks': blocks
                })
            
            doc.close()
            
            return {
                'pages': pages_content,
                'total_pages': len(pages_content),
                'extraction_method': 'PyMuPDF'
            }
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict[str, any]:
        """Fallback extraction with PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    pages_content.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'word_count': len(text.split())
                    })
                
                return {
                    'pages': pages_content,
                    'total_pages': len(pages_content),
                    'extraction_method': 'PyPDF2'
                }
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise

class LegalClauseExtractor:
    """RAG-based extraction of important legal clauses"""
    
    def __init__(self):
        self.encoder = SentenceTransformer('law-ai/InLegalBERT')
        self.clause_templates = self._load_clause_templates()
        self.importance_keywords = self._load_importance_keywords()
        
    def _load_clause_templates(self) -> Dict[str, List[str]]:
        """Load templates for different types of legal clauses"""
        return {
            'payment_terms': [
                "payment shall be made within",
                "fees are due",
                "compensation amount",
                "invoice payment terms"
            ],
            'termination_clauses': [
                "this agreement may be terminated",
                "termination for cause",
                "notice of termination",
                "effect of termination"
            ],
            'liability_limitations': [
                "limitation of liability",
                "damages shall not exceed",
                "liability is limited to",
                "exclusion of damages"
            ],
            'confidentiality': [
                "confidential information",
                "non-disclosure obligations",
                "proprietary information",
                "confidentiality period"
            ],
            'governing_law': [
                "governed by the laws of",
                "jurisdiction and venue",
                "applicable law",
                "dispute resolution"
            ],
            'force_majeure': [
                "force majeure event",
                "acts beyond reasonable control",
                "unforeseeable circumstances",
                "excused performance"
            ]
        }
    
    def _load_importance_keywords(self) -> Dict[str, float]:
        """Keywords that indicate clause importance"""
        return {
            'shall': 0.9,
            'must': 0.9,
            'required': 0.8,
            'liable': 0.8,
            'damages': 0.8,
            'terminate': 0.7,
            'breach': 0.8,
            'default': 0.7,
            'confidential': 0.7,
            'proprietary': 0.7,
            'indemnify': 0.8,
            'warranty': 0.7,
            'guarantee': 0.7,
            'penalty': 0.8,
            'liquidated damages': 0.9,
            'force majeure': 0.6,
            'governing law': 0.6,
            'jurisdiction': 0.6
        }
    
    def extract_clauses_with_rag(self, pdf_content: Dict, min_importance: float = 0.5) -> List[ExtractedClause]:
        """Extract important clauses using RAG approach"""
        all_clauses = []
        
        # Combine all text for context
        full_text = " ".join([page['text'] for page in pdf_content['pages']])
        
        for page in pdf_content['pages']:
            page_clauses = self._extract_clauses_from_page(
                page['text'], 
                page['page_number'], 
                full_text
            )
            all_clauses.extend(page_clauses)
        
        # Filter by importance and remove duplicates
        important_clauses = [c for c in all_clauses if c.importance_score >= min_importance]
        important_clauses = self._remove_duplicate_clauses(important_clauses)
        
        # Sort by importance
        important_clauses.sort(key=lambda x: x.importance_score, reverse=True)
        
        return important_clauses
    
    def _extract_clauses_from_page(self, page_text: str, page_num: int, full_context: str) -> List[ExtractedClause]:
        """Extract clauses from a single page"""
        clauses = []
        
        # Split into sentences/paragraphs
        sentences = self._split_into_legal_sentences(page_text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 50:  # Skip very short sentences
                continue
                
            clause_type = self._classify_clause_type(sentence)
            importance_score = self._calculate_importance_score(sentence)
            entities = self._extract_legal_entities(sentence)
            section = self._identify_section(sentence, page_text)
            
            if importance_score > 0.3:  # Minimum threshold
                clause = ExtractedClause(
                    text=sentence.strip(),
                    clause_type=clause_type,
                    importance_score=importance_score,
                    page_number=page_num,
                    section=section,
                    entities=entities,
                    context=self._get_surrounding_context(sentence, page_text)
                )
                clauses.append(clause)
        
        return clauses
    
    def _split_into_legal_sentences(self, text: str) -> List[str]:
        """Split text into legal sentences (handles legal formatting)"""
        # Handle legal numbering and formatting
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split on periods, but be careful with legal citations and abbreviations
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char == '.' and len(current_sentence) > 20:
                # Check if this is likely end of sentence
                next_chars = text[text.index(current_sentence) + len(current_sentence):text.index(current_sentence) + len(current_sentence) + 5]
                if re.match(r'\s+[A-Z]', next_chars) or not next_chars.strip():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if len(s.strip()) > 20]
    
    def _classify_clause_type(self, text: str) -> str:
        """Classify the type of legal clause"""
        text_lower = text.lower()
        
        # Use template matching
        best_match = "general"
        best_score = 0
        
        for clause_type, templates in self.clause_templates.items():
            for template in templates:
                if template.lower() in text_lower:
                    # Calculate similarity score
                    score = len(template) / len(text_lower)  # Simple scoring
                    if score > best_score:
                        best_score = score
                        best_match = clause_type
        
        return best_match
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score based on keywords and patterns"""
        text_lower = text.lower()
        score = 0.0
        
        # Keyword-based scoring
        for keyword, weight in self.importance_keywords.items():
            if keyword in text_lower:
                score += weight
        
        # Pattern-based scoring
        patterns = {
            r'shall\s+(?:not\s+)?(?:be|have|do|pay|provide)': 0.8,
            r'must\s+(?:not\s+)?(?:be|have|do|pay|provide)': 0.8,
            r'is\s+(?:not\s+)?(?:liable|responsible|required)': 0.7,
            r'in\s+the\s+event\s+of': 0.6,
            r'subject\s+to\s+the\s+terms': 0.5,
            r'notwithstanding\s+anything': 0.7,
            r'provided\s+that': 0.5,
            r'except\s+as\s+otherwise': 0.6
        }
        
        for pattern, weight in patterns.items():
            if re.search(pattern, text_lower):
                score += weight
        
        # Length penalty for very long clauses
        if len(text) > 500:
            score *= 0.8
        
        # Normalize score
        return min(1.0, score / 3.0)
    
    def _extract_legal_entities(self, text: str) -> List[Dict]:
        """Extract legal entities from text"""
        entities = []
        
        # Date patterns
        date_patterns = [
            (r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b', 'DATE'),
            (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'DATE'),
            (r'\b\d{1,2}\s+(?:days?|months?|years?)\b', 'DURATION')
        ]
        
        # Money patterns
        money_patterns = [
            (r'\$[\d,]+(?:\.\d{2})?', 'MONEY'),
            (r'\b\d+\s+dollars?\b', 'MONEY'),
            (r'\b(?:USD|EUR|GBP)\s*\d+', 'MONEY')
        ]
        
        # Legal citation patterns
        citation_patterns = [
            (r'\b\d+\s+U\.S\.\s+\d+', 'CITATION'),
            (r'\b\d+\s+F\.\d+d\s+\d+', 'CITATION'),
            (r'\b\d+\s+S\.Ct\.\s+\d+', 'CITATION')
        ]
        
        all_patterns = date_patterns + money_patterns + citation_patterns
        
        for pattern, entity_type in all_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'type': entity_type,
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    def _identify_section(self, sentence: str, page_text: str) -> str:
        """Identify which section of the document this sentence belongs to"""
        # Find the sentence position in the page
        sentence_pos = page_text.find(sentence)
        if sentence_pos == -1:
            return "unknown"
        
        # Look backwards for section headers
        preceding_text = page_text[:sentence_pos].lower()
        
        for section_name, pattern in {
            'definitions': r'(?i)(definitions?|defined terms?)',
            'obligations': r'(?i)(obligations?|duties)',
            'termination': r'(?i)(termination|end of agreement)',
            'payment': r'(?i)(payment|fees|compensation)',
            'liability': r'(?i)(liability|damages)',
            'dispute': r'(?i)(dispute resolution|arbitration)',
            'governing_law': r'(?i)(governing law|applicable law)',
            'confidentiality': r'(?i)(confidentiality|non-disclosure)'
        }.items():
            matches = list(re.finditer(pattern, preceding_text))
            if matches:
                return section_name
        
        return "general"
    
    def _get_surrounding_context(self, sentence: str, page_text: str, context_chars: int = 200) -> str:
        """Get surrounding context for the sentence"""
        sentence_pos = page_text.find(sentence)
        if sentence_pos == -1:
            return sentence
        
        start = max(0, sentence_pos - context_chars)
        end = min(len(page_text), sentence_pos + len(sentence) + context_chars)
        
        return page_text[start:end].strip()
    
    def _remove_duplicate_clauses(self, clauses: List[ExtractedClause]) -> List[ExtractedClause]:
        """Remove duplicate or very similar clauses"""
        if not clauses:
            return clauses
        
        # Encode all clause texts
        texts = [clause.text for clause in clauses]
        embeddings = self.encoder.encode(texts)
        
        # Build similarity index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype('float32'))
        
        # Find duplicates
        unique_clauses = []
        used_indices = set()
        
        for i, clause in enumerate(clauses):
            if i in used_indices:
                continue
            
            # Search for similar clauses
            query = embeddings[i:i+1]
            scores, indices = index.search(query.astype('float32'), len(clauses))
            
            # Mark similar clauses as used
            for score, idx in zip(scores[0], indices[0]):
                if score > 0.85 and idx != i:  # High similarity threshold
                    used_indices.add(idx)
            
            unique_clauses.append(clause)
            used_indices.add(i)
        
        return unique_clauses

class RAGClauseProcessor:
    """Main processor combining PDF extraction and RAG-based clause extraction"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.clause_extractor = LegalClauseExtractor()
    
    def process_pdf_document(self, pdf_path: str, min_importance: float = 0.5) -> Dict:
        """Process PDF and extract important clauses using RAG"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract text from PDF
            pdf_content = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Extract important clauses using RAG
            clauses = self.clause_extractor.extract_clauses_with_rag(
                pdf_content, 
                min_importance
            )
            
            # Organize results
            result = {
                'document_info': {
                    'total_pages': pdf_content['total_pages'],
                    'extraction_method': pdf_content['extraction_method'],
                    'total_clauses_found': len(clauses)
                },
                'important_clauses': [
                    {
                        'text': clause.text,
                        'type': clause.clause_type,
                        'importance_score': clause.importance_score,
                        'page_number': clause.page_number,
                        'section': clause.section,
                        'entities': clause.entities,
                        'context': clause.context
                    }
                    for clause in clauses
                ],
                'clause_summary': self._generate_clause_summary(clauses)
            }
            
            logger.info(f"Extracted {len(clauses)} important clauses from {pdf_content['total_pages']} pages")
            return result
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    def _generate_clause_summary(self, clauses: List[ExtractedClause]) -> Dict:
        """Generate summary statistics about extracted clauses"""
        if not clauses:
            return {}
        
        clause_types = {}
        sections = {}
        pages = {}
        
        for clause in clauses:
            clause_types[clause.clause_type] = clause_types.get(clause.clause_type, 0) + 1
            sections[clause.section] = sections.get(clause.section, 0) + 1
            pages[clause.page_number] = pages.get(clause.page_number, 0) + 1
        
        return {
            'total_clauses': len(clauses),
            'avg_importance_score': sum(c.importance_score for c in clauses) / len(clauses),
            'clause_types_distribution': clause_types,
            'sections_distribution': sections,
            'pages_with_clauses': len(pages),
            'most_important_clause': max(clauses, key=lambda x: x.importance_score).text[:100] + "..."
        }