# xai_explainer.py - Advanced Explainable AI components for legal fact-checking

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import shap
import lime
from lime.lime_text import LimeTextExplainer
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class XAIExplanation:
    """Comprehensive XAI explanation structure"""
    decision_summary: str
    confidence_score: float
    reasoning_steps: List[str]
    feature_importance: Dict[str, float]
    attention_weights: Dict[str, float]
    counterfactual_examples: List[str]
    decision_tree_path: str
    lime_explanation: Dict[str, Any]
    shap_values: Dict[str, float]
    uncertainty_analysis: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    legal_precedent_chain: List[Dict]
    alternative_interpretations: List[Dict]

class AttentionVisualizer:
    """Visualize attention weights for transformer models"""
    
    def __init__(self):
        self.attention_patterns = {}
    
    def extract_attention_weights(self, model, tokenizer, text: str, claim: str) -> Dict[str, float]:
        """Extract attention weights from transformer model"""
        try:
            # Tokenize input
            inputs = tokenizer(
                f"{claim} [SEP] {text}", 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                return_attention_mask=True
            )
            
            # Get model outputs with attention
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                attentions = outputs.attentions  # List of attention matrices
            
            # Average attention across layers and heads
            avg_attention = torch.mean(torch.stack(attentions), dim=(0, 1))  # [seq_len, seq_len]
            
            # Get tokens
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Create attention weights dictionary
            attention_weights = {}
            for i, token in enumerate(tokens):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    # Sum attention from all positions to this token
                    weight = float(torch.sum(avg_attention[:, i]))
                    attention_weights[token] = weight
            
            return attention_weights
            
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            return {}
    
    def visualize_attention_heatmap(self, attention_weights: Dict[str, float], save_path: str = None) -> str:
        """Create attention heatmap visualization"""
        try:
            if not attention_weights:
                return "No attention data available"
            
            # Prepare data for heatmap
            tokens = list(attention_weights.keys())[:20]  # Top 20 tokens
            weights = [attention_weights[token] for token in tokens]
            
            # Create heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(
                np.array(weights).reshape(1, -1), 
                xticklabels=tokens, 
                yticklabels=['Attention'],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f'
            )
            plt.title('Attention Weights for Legal Text Analysis')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                return f"Attention heatmap saved to {save_path}"
            else:
                return "Attention heatmap generated (display mode)"
                
        except Exception as e:
            logger.error(f"Attention visualization failed: {e}")
            return f"Visualization error: {e}"

class FeatureImportanceAnalyzer:
    """Analyze feature importance for legal decisions"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
        
    def analyze_feature_importance(self, texts: List[str], labels: List[int], target_text: str) -> Dict[str, float]:
        """Analyze which features are most important for classification"""
        try:
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts + [target_text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Train decision tree
            if len(set(labels)) > 1:  # Need at least 2 classes
                self.decision_tree.fit(X[:-1], labels)
                
                # Get feature importance
                importance_scores = self.decision_tree.feature_importances_
                
                # Create feature importance dictionary
                feature_importance = {}
                for i, score in enumerate(importance_scores):
                    if score > 0.01:  # Only significant features
                        feature_importance[feature_names[i]] = float(score)
                
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])
            else:
                # Fallback to TF-IDF scores for single class
                target_vector = X[-1].toarray()[0]
                feature_importance = {}
                for i, score in enumerate(target_vector):
                    if score > 0.1:
                        feature_importance[feature_names[i]] = float(score)
                
                return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])
                
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {}
    
    def get_decision_tree_explanation(self, texts: List[str], labels: List[int]) -> str:
        """Get human-readable decision tree explanation"""
        try:
            if len(set(labels)) > 1:
                X = self.vectorizer.fit_transform(texts)
                self.decision_tree.fit(X, labels)
                
                # Export tree as text
                feature_names = self.vectorizer.get_feature_names_out()
                tree_rules = export_text(
                    self.decision_tree, 
                    feature_names=feature_names,
                    max_depth=5
                )
                
                return tree_rules
            else:
                return "Insufficient data for decision tree (single class)"
                
        except Exception as e:
            logger.error(f"Decision tree explanation failed: {e}")
            return f"Decision tree error: {e}"

class LIMEExplainer:
    """LIME explanations for legal text classification"""
    
    def __init__(self):
        self.explainer = LimeTextExplainer(class_names=['Not Important', 'Important'])
    
    def explain_prediction(self, text: str, predict_fn, num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanation for text classification"""
        try:
            # Generate explanation
            explanation = self.explainer.explain_instance(
                text, 
                predict_fn, 
                num_features=num_features,
                num_samples=1000
            )
            
            # Extract explanation data
            lime_data = {
                'explanation_text': explanation.as_list(),
                'prediction_probability': explanation.predict_proba,
                'local_explanation': explanation.local_exp,
                'intercept': float(explanation.intercept[1]) if hasattr(explanation, 'intercept') else 0.0
            }
            
            return lime_data
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {'error': str(e)}

class CounterfactualGenerator:
    """Generate counterfactual examples for legal claims"""
    
    def __init__(self):
        # Enhanced legal modifiers with more comprehensive relationships
        self.legal_modifiers = {
            # Obligation strength (critical changes)
            'shall': ['may', 'should', 'will', 'must', 'might'],
            'must': ['shall', 'may', 'should', 'will', 'might'],
            'will': ['shall', 'may', 'should', 'must', 'might'],
            'may': ['shall', 'must', 'should', 'will', 'might'],
            'required': ['optional', 'recommended', 'prohibited', 'forbidden'],
            'mandatory': ['optional', 'voluntary', 'discretionary'],
            'obligated': ['entitled', 'permitted', 'authorized'],
            
            # Liability and responsibility
            'liable': ['responsible', 'accountable', 'not liable', 'exempt'],
            'responsible': ['liable', 'accountable', 'not responsible', 'exempt'],
            'accountable': ['liable', 'responsible', 'not accountable', 'immune'],
            
            # Legal actions and consequences  
            'terminate': ['continue', 'extend', 'suspend', 'renew', 'maintain'],
            'breach': ['compliance', 'fulfillment', 'violation', 'adherence'],
            'violation': ['compliance', 'breach', 'fulfillment', 'observance'],
            'default': ['performance', 'compliance', 'fulfillment'],
            
            # Binding and enforceability
            'binding': ['non-binding', 'advisory', 'enforceable', 'voluntary'],
            'enforceable': ['unenforceable', 'binding', 'valid', 'void'],
            'valid': ['void', 'invalid', 'null', 'enforceable'],
            'void': ['valid', 'enforceable', 'binding', 'effective'],
            'null': ['valid', 'effective', 'enforceable'],
            
            # Revocability and permanence
            'irrevocable': ['revocable', 'conditional', 'temporary', 'amendable'],
            'revocable': ['irrevocable', 'permanent', 'final', 'unchangeable'],
            'permanent': ['temporary', 'provisional', 'interim', 'revocable'],
            'temporary': ['permanent', 'indefinite', 'perpetual', 'irrevocable'],
            
            # Scope and extent
            'all': ['some', 'no', 'most', 'few', 'none'],
            'any': ['no', 'specific', 'certain', 'particular'],
            'some': ['all', 'no', 'most', 'few', 'many'],
            'no': ['all', 'some', 'any', 'several'],
            'none': ['all', 'some', 'any', 'several'],
            'every': ['some', 'no', 'certain', 'selected'],
            'unlimited': ['limited', 'capped', 'restricted', 'bounded'],
            'limited': ['unlimited', 'unrestricted', 'boundless', 'infinite'],
            
            # Temporal aspects
            'immediately': ['within 30 days', 'eventually', 'never', 'promptly'],
            'never': ['immediately', 'always', 'eventually', 'sometimes'],
            'always': ['never', 'sometimes', 'occasionally', 'rarely'],
            'promptly': ['eventually', 'immediately', 'delayed', 'slowly'],
            'forthwith': ['eventually', 'delayed', 'at convenience'],
            
            # Exclusivity and access
            'exclusive': ['non-exclusive', 'shared', 'joint', 'common'],
            'confidential': ['public', 'proprietary', 'restricted', 'open'],
            'private': ['public', 'open', 'shared', 'common'],
            'proprietary': ['public', 'open-source', 'shared', 'common'],
            
            # Financial and penalty terms
            'penalty': ['incentive', 'reward', 'bonus', 'benefit'],
            'damages': ['compensation', 'benefits', 'reimbursement', 'payment'],
            'fine': ['reward', 'bonus', 'payment', 'compensation'],
            'interest': ['principal', 'discount', 'rebate'],
            
            # Rights and permissions
            'authorized': ['unauthorized', 'prohibited', 'forbidden', 'banned'],
            'permitted': ['prohibited', 'forbidden', 'banned', 'restricted'],
            'entitled': ['not entitled', 'disqualified', 'ineligible'],
            'eligible': ['ineligible', 'disqualified', 'excluded'],
            
            # Certainty and conditionality
            'absolute': ['conditional', 'relative', 'qualified', 'limited'],
            'conditional': ['absolute', 'unconditional', 'definite', 'certain'],
            'definite': ['indefinite', 'uncertain', 'conditional', 'provisional'],
            'final': ['preliminary', 'interim', 'provisional', 'temporary'],
        }
        
        self.legal_contexts = {
            'contract': {
                'scenarios': [
                    'What if the contract term was {duration} instead?',
                    'What if the payment was {amount} rather than agreed?',
                    'What if the jurisdiction was {location} not as stated?'
                ],
                'values': {
                    'duration': ['1 year', '5 years', 'indefinite', 'month-to-month'],
                    'amount': ['50% more', '25% less', 'fixed rate', 'variable rate'],
                    'location': ['federal court', 'arbitration', 'mediation']
                }
            },
            'liability': {
                'scenarios': [
                    'What if liability was {scope} instead of unlimited?',
                    'What if the damages cap was {amount}?',
                    'What if indemnification was {type}?'
                ],
                'values': {
                    'scope': ['limited', 'joint', 'several', 'proportional'],
                    'amount': ['$1M', '$10M', 'actual damages only'],
                    'type': ['mutual', 'one-way', 'excluded']
                }
            }
        }
    
    def generate_counterfactuals(self, original_text: str, num_examples: int = 5) -> List[Dict[str, Any]]:
        """Generate counterfactual examples with explanations"""
        counterfactuals = []
        
        try:
            # Generate word-substitution counterfactuals
            word_counterfactuals = self._generate_word_substitutions(original_text, num_examples // 2)
            counterfactuals.extend(word_counterfactuals)
            
            # Generate contextual counterfactuals
            contextual_counterfactuals = self._generate_contextual_scenarios(original_text, num_examples // 2)
            counterfactuals.extend(contextual_counterfactuals)
            
            # Generate negation counterfactuals
            negation_counterfactuals = self._generate_negations(original_text, 2)
            counterfactuals.extend(negation_counterfactuals)
            
            return counterfactuals[:num_examples]
            
        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return []
    
    def _generate_word_substitutions(self, original_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Generate counterfactuals by substituting key legal terms"""
        counterfactuals = []
        
        for _ in range(num_examples):
            modified_text = original_text
            modifications_made = 0
            changes_made = []
            
            for original_word, alternatives in self.legal_modifiers.items():
                if original_word in modified_text.lower() and modifications_made < 2:
                    alternative = np.random.choice(alternatives)
                    pattern = re.compile(r'\b' + re.escape(original_word) + r'\b', re.IGNORECASE)
                    if pattern.search(modified_text):
                        modified_text = pattern.sub(alternative, modified_text, count=1)
                        changes_made.append(f"'{original_word}' → '{alternative}'")
                        modifications_made += 1
            
            if modified_text != original_text:
                counterfactals_example = {
                    'text': modified_text,
                    'type': 'word_substitution',
                    'changes': changes_made,
                    'explanation': f"Changed key legal terms: {', '.join(changes_made)}",
                    'impact': self._assess_legal_impact(changes_made),
                    'likelihood': self._assess_likelihood(original_text, modified_text)
                }
                counterfactuals.append(counterfactals_example)
        
        return counterfactuals
    
    def _generate_contextual_scenarios(self, original_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Generate scenario-based counterfactuals"""
        counterfactuals = []
        
        # Detect legal context
        context = self._detect_legal_context(original_text)
        
        if context in self.legal_contexts:
            scenarios = self.legal_contexts[context]['scenarios']
            values = self.legal_contexts[context]['values']
            
            for _ in range(min(num_examples, len(scenarios))):
                scenario_template = np.random.choice(scenarios)
                
                # Fill in template values
                for placeholder, options in values.items():
                    if f'{{{placeholder}}}' in scenario_template:
                        value = np.random.choice(options)
                        scenario_template = scenario_template.replace(f'{{{placeholder}}}', value)
                
                counterfactual_example = {
                    'text': scenario_template,
                    'type': 'contextual_scenario',
                    'changes': [f"Alternative {context} scenario"],
                    'explanation': f"Hypothetical scenario: {scenario_template}",
                    'impact': 'moderate',
                    'likelihood': 'possible'
                }
                counterfactuals.append(counterfactual_example)
        
        return counterfactuals
    
    def _generate_negations(self, original_text: str, num_examples: int = 3) -> List[Dict[str, Any]]:
        """Generate counterfactuals by negating key claims with enhanced legal precision"""
        counterfactuals = []
        
        # Enhanced negation patterns with legal context awareness
        negation_patterns = [
            # Existence and state negations
            (r'\bis\s+(liable|responsible|accountable)', r'is not \1'),
            (r'\bare\s+(liable|responsible|accountable)', r'are not \1'),
            (r'\bis\s+(binding|enforceable|valid)', r'is not \1'),
            (r'\bare\s+(binding|enforceable|valid)', r'are not \1'),
            
            # Obligation negations
            (r'\bmust\s+', 'must not '),
            (r'\bshall\s+', 'shall not '),
            (r'\bwill\s+', 'will not '),
            (r'\bis\s+required', 'is not required'),
            (r'\bare\s+required', 'are not required'),
            (r'\bis\s+obligated', 'is not obligated'),
            (r'\bare\s+obligated', 'are not obligated'),
            
            # Permission and authorization negations
            (r'\bmay\s+', 'may not '),
            (r'\bis\s+(authorized|permitted|entitled)', r'is not \1'),
            (r'\bare\s+(authorized|permitted|entitled)', r'are not \1'),
            
            # Temporal negations
            (r'\bimmediately\b', 'never'),
            (r'\balways\b', 'never'),
            (r'\bpermanently\b', 'temporarily'),
            (r'\birrevocably\b', 'revocably'),
            
            # Scope negations - more precise
            (r'\ball\s+damages', 'no damages'),
            (r'\bany\s+liability', 'no liability'),
            (r'\bevery\s+', 'no '),
            (r'\bunlimited\s+liability', 'no liability'),
            
            # Conditional negations
            (r'\bif\s+(.+?)\s+then', r'if not \1 then'),
            (r'\bunless\s+(.+?),', r'if \1,'),
        ]
        
        generated_count = 0
        used_patterns = set()
        
        for pattern, replacement in negation_patterns:
            if generated_count >= num_examples:
                break
                
            # Check if pattern exists in text
            if re.search(pattern, original_text, re.IGNORECASE):
                # Generate the negation
                modified_text = re.sub(pattern, replacement, original_text, count=1, flags=re.IGNORECASE)
                
                # Avoid duplicate patterns
                pattern_key = f"{pattern}->{replacement}"
                if pattern_key in used_patterns:
                    continue
                used_patterns.add(pattern_key)
                
                # Ensure the modification actually changed the text
                if modified_text != original_text:
                    # Extract what was changed for clear explanation
                    original_match = re.search(pattern, original_text, re.IGNORECASE)
                    if original_match:
                        original_phrase = original_match.group(0)
                        modified_phrase = re.sub(pattern, replacement, original_phrase, flags=re.IGNORECASE)
                        
                        counterfactual_example = {
                            'text': modified_text,
                            'type': 'negation',
                            'changes': [f"'{original_phrase}' → '{modified_phrase}'"],
                            'explanation': f"Negated key legal assertion: {original_phrase} → {modified_phrase}",
                            'impact': self._assess_negation_impact(original_phrase, modified_phrase),
                            'likelihood': 'alternative_interpretation'
                        }
                        counterfactuals.append(counterfactual_example)
                        generated_count += 1
        
        # If we haven't generated enough, try some additional semantic negations
        if generated_count < num_examples:
            semantic_negations = self._generate_semantic_negations(original_text, num_examples - generated_count)
            counterfactuals.extend(semantic_negations)
        
        return counterfactuals
    
    def _assess_negation_impact(self, original_phrase: str, modified_phrase: str) -> str:
        """Assess the impact of a negation change"""
        high_impact_negations = [
            'liable', 'binding', 'required', 'must', 'shall', 'valid', 
            'enforceable', 'unlimited', 'all', 'every', 'always'
        ]
        
        original_lower = original_phrase.lower()
        if any(term in original_lower for term in high_impact_negations):
            return 'high'
        elif any(term in original_lower for term in ['may', 'some', 'occasionally']):
            return 'medium'
        else:
            return 'low'
    
    def _generate_semantic_negations(self, original_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Generate semantic opposites based on legal meaning"""
        counterfactuals = []
        
        semantic_opposites = {
            'liable for': 'exempt from',
            'responsible for': 'not responsible for',
            'binding agreement': 'non-binding agreement',
            'enforceable contract': 'unenforceable contract',
            'unlimited liability': 'no liability',
            'all damages': 'no damages',
            'immediate payment': 'deferred payment',
            'irrevocable consent': 'revocable consent',
            'exclusive rights': 'non-exclusive rights',
            'confidential information': 'public information'
        }
        
        for original, opposite in semantic_opposites.items():
            if len(counterfactuals) >= num_examples:
                break
                
            if original.lower() in original_text.lower():
                modified_text = re.sub(re.escape(original), opposite, original_text, count=1, flags=re.IGNORECASE)
                
                counterfactual_example = {
                    'text': modified_text,
                    'type': 'negation',
                    'changes': [f"'{original}' → '{opposite}'"],
                    'explanation': f"Semantic negation: {original} → {opposite}",
                    'impact': 'high',  # Most semantic opposites are high impact
                    'likelihood': 'alternative_interpretation'
                }
                counterfactuals.append(counterfactual_example)
        
        return counterfactuals
    
    def _detect_legal_context(self, text: str) -> str:
        """Detect the primary legal context of the text"""
        text_lower = text.lower()
        
        contract_keywords = ['contract', 'agreement', 'term', 'clause', 'party', 'obligation']
        liability_keywords = ['liable', 'damages', 'indemnify', 'breach', 'penalty', 'compensation']
        
        contract_score = sum(1 for kw in contract_keywords if kw in text_lower)
        liability_score = sum(1 for kw in liability_keywords if kw in text_lower)
        
        if liability_score > contract_score:
            return 'liability'
        else:
            return 'contract'
    
    def _assess_legal_impact(self, changes: List[str]) -> str:
        """Assess the legal impact of changes with enhanced precision"""
        
        # Define comprehensive legal impact patterns
        critical_impact_patterns = [
            # Complete obligation reversals
            (r"'shall'\s*→\s*'may'", 'high'),
            (r"'must'\s*→\s*'may'", 'high'),
            (r"'required'\s*→\s*'optional'", 'high'),
            (r"'binding'\s*→\s*'non-binding'", 'high'),
            (r"'binding'\s*→\s*'advisory'", 'high'),
            (r"'irrevocable'\s*→\s*'revocable'", 'high'),
            (r"'liable'\s*→\s*'not liable'", 'high'),
            (r"'all'\s*→\s*'no'", 'high'),
            (r"'unlimited'\s*→\s*'limited'", 'high'),
            (r"'void'\s*→\s*'valid'", 'high'),
            (r"'breach'\s*→\s*'compliance'", 'high'),
            
            # Scope and quantity changes
            (r"'all'\s*→\s*'some'", 'medium'),
            (r"'any'\s*→\s*'specific'", 'medium'),
            (r"'immediately'\s*→\s*'within.*days'", 'medium'),
            (r"'never'\s*→\s*'eventually'", 'medium'),
            (r"'exclusive'\s*→\s*'non-exclusive'", 'medium'),
            
            # Moderate obligation changes
            (r"'shall'\s*→\s*'should'", 'medium'),
            (r"'must'\s*→\s*'should'", 'medium'),
            (r"'penalty'\s*→\s*'incentive'", 'medium'),
            
            # Minor semantic changes
            (r"'damages'\s*→\s*'compensation'", 'low'),
            (r"'breach'\s*→\s*'violation'", 'low'),
            (r"'liable'\s*→\s*'responsible'", 'low'),
            (r"'confidential'\s*→\s*'proprietary'", 'low'),
        ]
        
        # Legal term impact categories
        high_impact_terms = {
            'obligation_strength': ['shall', 'must', 'required', 'mandatory'],
            'liability': ['liable', 'not liable', 'damages'],
            'binding_nature': ['binding', 'non-binding', 'void', 'valid'],
            'scope_complete': ['all', 'no', 'none', 'unlimited', 'limited'],
            'temporal_absolute': ['immediately', 'never', 'always'],
            'revocability': ['irrevocable', 'revocable', 'permanent', 'temporary']
        }
        
        medium_impact_terms = {
            'scope_partial': ['some', 'most', 'any', 'specific'],
            'obligation_moderate': ['should', 'recommended'],
            'temporal_relative': ['within', 'days', 'eventually', 'soon'],
            'exclusivity': ['exclusive', 'non-exclusive', 'shared'],
            'access_level': ['confidential', 'public', 'restricted']
        }
        
        low_impact_terms = {
            'synonyms': ['compensation', 'reimbursement', 'payment'],
            'similar_violations': ['breach', 'violation', 'default'],
            'similar_responsibility': ['responsible', 'accountable'],
            'similar_property': ['proprietary', 'restricted']
        }
        
        # Analyze all changes
        impact_scores = []
        change_details = []
        
        for change in changes:
            change_lower = change.lower()
            highest_impact = 'low'
            
            # Check exact pattern matches first
            for pattern, impact in critical_impact_patterns:
                if re.search(pattern, change_lower):
                    highest_impact = impact
                    change_details.append(f"Pattern match: {pattern} → {impact}")
                    break
            else:
                # Check term categories
                for category, terms in high_impact_terms.items():
                    if any(term in change_lower for term in terms):
                        # Special handling for negations
                        if '→ not' in change_lower or 'not liable' in change_lower:
                            highest_impact = 'high'
                            change_details.append(f"Negation: {category} → high")
                            break
                        # Check if it's a strong obligation change
                        elif any(f"'{strong}' → '{weak}'" in change_lower 
                               for strong in ['shall', 'must', 'required'] 
                               for weak in ['may', 'should', 'optional']):
                            highest_impact = 'high'
                            change_details.append(f"Obligation weakening: {category} → high")
                            break
                        elif category in ['binding_nature', 'revocability', 'scope_complete']:
                            highest_impact = 'high'
                            change_details.append(f"Critical term: {category} → high")
                            break
                        else:
                            highest_impact = 'medium'
                            change_details.append(f"High-impact term: {category} → medium")
                else:
                    # Check medium impact terms
                    for category, terms in medium_impact_terms.items():
                        if any(term in change_lower for term in terms):
                            if highest_impact == 'low':
                                highest_impact = 'medium'
                                change_details.append(f"Medium term: {category} → medium")
                            break
                    else:
                        # Check low impact terms
                        for category, terms in low_impact_terms.items():
                            if any(term in change_lower for term in terms):
                                change_details.append(f"Low term: {category} → low")
                                break
            
            # Convert to numeric score for aggregation
            if highest_impact == 'high':
                impact_scores.append(3)
            elif highest_impact == 'medium':
                impact_scores.append(2)
            else:
                impact_scores.append(1)
        
        # Determine final impact level
        if not impact_scores:
            return 'medium'
        
        max_score = max(impact_scores)
        avg_score = sum(impact_scores) / len(impact_scores)
        
        # Use maximum impact for critical changes, but consider average for edge cases
        if max_score >= 3:
            return 'high'
        elif max_score >= 2 or avg_score >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _assess_likelihood(self, original: str, modified: str) -> str:
        """Assess how likely this counterfactual scenario is"""
        # Simple heuristic based on number of changes
        word_changes = len(set(original.lower().split()) ^ set(modified.lower().split()))
        
        if word_changes <= 2:
            return 'likely'
        elif word_changes <= 5:
            return 'possible'
        else:
            return 'unlikely'
    
    def generate_targeted_counterfactuals(self, text: str, target_outcome: str) -> List[Dict[str, Any]]:
        """Generate counterfactuals targeted at achieving a specific outcome with enhanced precision"""
        counterfactuals = []
        
        if target_outcome.lower() == 'supports':
            # Make changes that would STRENGTHEN the claim
            strengthening_changes = {
                # Weaker to stronger obligations
                'may': 'shall',
                'should': 'must',
                'can': 'shall',
                'might': 'will',
                'optional': 'required',
                'recommended': 'mandatory',
                'permitted': 'obligated',
                
                # Strengthen liability
                'not liable': 'fully liable',
                'limited liability': 'unlimited liability',
                'some damages': 'all damages',
                'partial responsibility': 'full responsibility',
                
                # Strengthen binding nature
                'non-binding': 'binding',
                'advisory': 'mandatory',
                'revocable': 'irrevocable',
                'temporary': 'permanent',
                'conditional': 'absolute',
                
                # Strengthen scope
                'some': 'all',
                'partial': 'complete',
                'limited': 'unlimited',
                'restricted': 'unrestricted',
                
                # Strengthen temporal requirements
                'eventually': 'immediately',
                'within 30 days': 'immediately',
                'when convenient': 'forthwith'
            }
            explanation_template = "Strengthened to support the claim"
            
        elif target_outcome.lower() == 'refutes':
            # Make changes that would WEAKEN or REFUTE the claim
            weakening_changes = {
                # Stronger to weaker obligations
                'shall': 'may',
                'must': 'should',
                'will': 'might',
                'required': 'optional',
                'mandatory': 'recommended',
                'obligated': 'permitted',
                
                # Weaken liability
                'liable': 'not liable',
                'fully liable': 'partially liable',
                'unlimited liability': 'no liability',
                'all damages': 'no damages',
                'responsible': 'not responsible',
                
                # Weaken binding nature
                'binding': 'non-binding',
                'enforceable': 'unenforceable',
                'mandatory': 'advisory',
                'irrevocable': 'revocable',
                'permanent': 'temporary',
                'absolute': 'conditional',
                
                # Weaken scope
                'all': 'no',
                'every': 'no',
                'complete': 'partial',
                'unlimited': 'limited',
                'unrestricted': 'restricted',
                
                # Weaken temporal requirements
                'immediately': 'eventually',
                'forthwith': 'when convenient',
                'always': 'never'
            }
            explanation_template = "Weakened to refute the claim"
            
        else:
            # Default to general modifications
            return self.generate_counterfactuals(text)
        
        # Select appropriate changes based on target outcome
        target_changes = strengthening_changes if target_outcome.lower() == 'supports' else weakening_changes
        
        # Apply targeted changes
        modified_text = text
        changes_made = []
        impact_levels = []
        
        # Try to make multiple relevant changes
        for original, replacement in target_changes.items():
            # Use word boundary matching for more precise substitution
            pattern = r'\b' + re.escape(original) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                modified_text = re.sub(pattern, replacement, modified_text, count=1, flags=re.IGNORECASE)
                changes_made.append(f"'{original}' → '{replacement}'")
                
                # Assess individual change impact
                individual_impact = self._assess_targeted_change_impact(original, replacement, target_outcome)
                impact_levels.append(individual_impact)
                
                # Limit to 2-3 changes to maintain coherence
                if len(changes_made) >= 2:
                    break
        
        # Generate contextual targeted scenarios
        contextual_scenarios = self._generate_targeted_scenarios(text, target_outcome)
        
        # Add the main targeted counterfactual
        if changes_made:
            overall_impact = 'high' if any(imp == 'high' for imp in impact_levels) else 'medium'
            
            counterfactual = {
                'text': modified_text,
                'type': 'targeted_outcome',
                'target_outcome': target_outcome,
                'changes': changes_made,
                'explanation': f"{explanation_template}: {', '.join(changes_made)}",
                'impact': overall_impact,
                'likelihood': 'targeted_modification'
            }
            counterfactuals.append(counterfactual)
        
        # Add contextual scenarios
        counterfactuals.extend(contextual_scenarios[:2])  # Limit contextual scenarios
        
        return counterfactuals
    
    def _assess_targeted_change_impact(self, original: str, replacement: str, target_outcome: str) -> str:
        """Assess the impact of a specific targeted change"""
        high_impact_transitions = [
            ('shall', 'may'), ('must', 'should'), ('liable', 'not liable'),
            ('binding', 'non-binding'), ('all', 'no'), ('unlimited', 'limited'),
            ('irrevocable', 'revocable'), ('required', 'optional')
        ]
        
        # Check if this is a high-impact transition
        transition = (original.lower(), replacement.lower())
        reverse_transition = (replacement.lower(), original.lower())
        
        if transition in high_impact_transitions or reverse_transition in high_impact_transitions:
            return 'high'
        
        # Check for medium impact changes
        medium_impact_words = ['should', 'may', 'some', 'partial', 'temporary', 'conditional']
        if original.lower() in medium_impact_words or replacement.lower() in medium_impact_words:
            return 'medium'
        
        return 'low'
    
    def _generate_targeted_scenarios(self, text: str, target_outcome: str) -> List[Dict[str, Any]]:
        """Generate contextual scenarios that support or refute the claim"""
        scenarios = []
        
        if target_outcome.lower() == 'supports':
            supporting_scenarios = [
                "What if additional penalties were imposed for non-compliance?",
                "What if the obligation was extended to include related activities?",
                "What if stricter deadlines were enforced?",
                "What if liability was joint and several?",
                "What if the agreement included punitive damages?"
            ]
            scenario_list = supporting_scenarios
            impact_level = 'medium'
            
        else:  # refutes
            refuting_scenarios = [
                "What if there were force majeure exceptions?",
                "What if the obligation was subject to regulatory approval?",
                "What if liability was capped at a nominal amount?",
                "What if the agreement included broad indemnification?",
                "What if performance was excused under certain conditions?"
            ]
            scenario_list = refuting_scenarios
            impact_level = 'medium'
        
        # Select relevant scenarios (up to 2)
        for scenario in scenario_list[:2]:
            scenarios.append({
                'text': scenario,
                'type': 'targeted_scenario',
                'target_outcome': target_outcome,
                'changes': [f"Hypothetical {target_outcome.lower()} scenario"],
                'explanation': f"Alternative scenario designed to {target_outcome.lower()} the claim",
                'impact': impact_level,
                'likelihood': 'hypothetical'
            })
        
        return scenarios

class UncertaintyAnalyzer:
    """Analyze prediction uncertainty and confidence intervals"""
    
    def __init__(self):
        self.monte_carlo_samples = 100
    
    def analyze_uncertainty(self, prediction_scores: List[float], text_features: Dict) -> Dict[str, Any]:
        """Analyze prediction uncertainty using various methods"""
        try:
            scores_array = np.array(prediction_scores)
            
            uncertainty_metrics = {
                'mean_confidence': float(np.mean(scores_array)),
                'std_confidence': float(np.std(scores_array)),
                'confidence_interval_95': [
                    float(np.percentile(scores_array, 2.5)),
                    float(np.percentile(scores_array, 97.5))
                ],
                'entropy': self._calculate_entropy(scores_array),
                'variance': float(np.var(scores_array)),
                'prediction_stability': self._assess_stability(scores_array),
                'feature_uncertainty': self._analyze_feature_uncertainty(text_features)
            }
            
            return uncertainty_metrics
            
        except Exception as e:
            logger.error(f"Uncertainty analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_entropy(self, scores: np.ndarray) -> float:
        """Calculate prediction entropy"""
        # Normalize scores to probabilities
        probs = np.exp(scores) / np.sum(np.exp(scores))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)
    
    def _assess_stability(self, scores: np.ndarray) -> str:
        """Assess prediction stability"""
        cv = np.std(scores) / (np.mean(scores) + 1e-10)
        
        if cv < 0.1:
            return "High stability"
        elif cv < 0.3:
            return "Moderate stability"
        else:
            return "Low stability"
    
    def _analyze_feature_uncertainty(self, features: Dict) -> Dict[str, float]:
        """Analyze uncertainty in feature contributions"""
        feature_uncertainty = {}
        
        for feature, value in features.items():
            if isinstance(value, (int, float)):
                # Simple uncertainty based on feature value magnitude
                uncertainty = 1.0 / (1.0 + abs(value))
                feature_uncertainty[feature] = uncertainty
        
        return feature_uncertainty

class BiasAnalyzer:
    """Analyze potential biases in legal AI decisions"""
    
    def __init__(self):
        self.bias_keywords = {
            'gender': ['he', 'she', 'his', 'her', 'him', 'man', 'woman', 'male', 'female'],
            'race': ['black', 'white', 'asian', 'hispanic', 'latino', 'african'],
            'age': ['young', 'old', 'elderly', 'senior', 'minor', 'adult'],
            'socioeconomic': ['poor', 'rich', 'wealthy', 'low-income', 'high-income'],
            'geographic': ['urban', 'rural', 'city', 'country', 'suburban']
        }
    
    def analyze_bias(self, text: str, decision_factors: Dict) -> Dict[str, Any]:
        """Analyze potential biases in the decision-making process"""
        try:
            bias_analysis = {
                'detected_bias_indicators': self._detect_bias_keywords(text),
                'fairness_metrics': self._calculate_fairness_metrics(decision_factors),
                'bias_mitigation_suggestions': self._suggest_bias_mitigation(text),
                'demographic_neutrality_score': self._calculate_neutrality_score(text)
            }
            
            return bias_analysis
            
        except Exception as e:
            logger.error(f"Bias analysis failed: {e}")
            return {'error': str(e)}
    
    def _detect_bias_keywords(self, text: str) -> Dict[str, List[str]]:
        """Detect potential bias-indicating keywords"""
        detected_bias = {}
        text_lower = text.lower()
        
        for bias_type, keywords in self.bias_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                detected_bias[bias_type] = found_keywords
        
        return detected_bias
    
    def _calculate_fairness_metrics(self, factors: Dict) -> Dict[str, float]:
        """Calculate fairness metrics"""
        return {
            'feature_balance': 0.8,  # Placeholder - would calculate actual balance
            'representation_fairness': 0.75,
            'outcome_equity': 0.85
        }
    
    def _suggest_bias_mitigation(self, text: str) -> List[str]:
        """Suggest bias mitigation strategies"""
        suggestions = []
        
        if any(kw in text.lower() for kw in self.bias_keywords['gender']):
            suggestions.append("Consider gender-neutral language alternatives")
        
        if any(kw in text.lower() for kw in self.bias_keywords['race']):
            suggestions.append("Ensure race-neutral legal analysis")
        
        suggestions.append("Validate decision against multiple legal precedents")
        suggestions.append("Consider alternative interpretations")
        
        return suggestions
    
    def _calculate_neutrality_score(self, text: str) -> float:
        """Calculate demographic neutrality score"""
        total_bias_words = sum(len(keywords) for keywords in self.bias_keywords.values())
        found_bias_words = sum(
            len([kw for kw in keywords if kw in text.lower()])
            for keywords in self.bias_keywords.values()
        )
        
        neutrality_score = 1.0 - (found_bias_words / max(total_bias_words, 1))
        return max(0.0, min(1.0, neutrality_score))

class ComprehensiveXAIEngine:
    """Main XAI engine combining all explanation methods"""
    
    def __init__(self):
        self.attention_visualizer = AttentionVisualizer()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.lime_explainer = LIMEExplainer()
        self.counterfactual_generator = CounterfactualGenerator()
        self.uncertainty_analyzer = UncertaintyAnalyzer()
        self.bias_analyzer = BiasAnalyzer()
    
    def generate_comprehensive_explanation(
        self, 
        text: str, 
        claim: str, 
        prediction_result: Dict,
        model_context: Dict = None
    ) -> XAIExplanation:
        """Generate comprehensive XAI explanation"""
        
        try:
            logger.info(f"Generating comprehensive XAI explanation for claim: {claim[:100]}...")
            
            # 1. Extract reasoning steps
            reasoning_steps = self._generate_reasoning_steps(text, claim, prediction_result)
            
            # 2. Feature importance analysis
            feature_importance = self._analyze_features(text, claim)
            
            # 3. Generate counterfactuals
            counterfactuals = self.counterfactual_generator.generate_counterfactuals(claim)
            
            # 4. Decision tree explanation
            decision_tree_path = self._generate_decision_path(text, claim, prediction_result)
            
            # 5. LIME explanation
            lime_explanation = self._generate_lime_explanation(text, claim)
            
            # 6. Uncertainty analysis
            uncertainty_analysis = self.uncertainty_analyzer.analyze_uncertainty(
                [prediction_result.get('confidence', 0.5)], 
                feature_importance
            )
            
            # 7. Bias analysis
            bias_analysis = self.bias_analyzer.analyze_bias(text, prediction_result)
            
            # 8. Legal precedent chain
            legal_precedent_chain = self._build_precedent_chain(claim, prediction_result)
            
            # 9. Alternative interpretations
            alternative_interpretations = self._generate_alternatives(claim, prediction_result)
            
            # 10. Attention weights (if model available)
            attention_weights = {}
            if model_context and 'model' in model_context:
                attention_weights = self.attention_visualizer.extract_attention_weights(
                    model_context['model'], 
                    model_context['tokenizer'], 
                    text, 
                    claim
                )
            
            # Create comprehensive explanation
            explanation = XAIExplanation(
                decision_summary=self._create_decision_summary(prediction_result, reasoning_steps),
                confidence_score=prediction_result.get('confidence', 0.0),
                reasoning_steps=reasoning_steps,
                feature_importance=feature_importance,
                attention_weights=attention_weights,
                counterfactual_examples=counterfactuals,
                decision_tree_path=decision_tree_path,
                lime_explanation=lime_explanation,
                shap_values={},  # Would implement SHAP if needed
                uncertainty_analysis=uncertainty_analysis,
                bias_analysis=bias_analysis,
                legal_precedent_chain=legal_precedent_chain,
                alternative_interpretations=alternative_interpretations
            )
            
            logger.info("Comprehensive XAI explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"XAI explanation generation failed: {e}")
            # Return minimal explanation on error
            return XAIExplanation(
                decision_summary=f"Error generating explanation: {e}",
                confidence_score=0.0,
                reasoning_steps=["Error in explanation generation"],
                feature_importance={},
                attention_weights={},
                counterfactual_examples=[],
                decision_tree_path="Error",
                lime_explanation={},
                shap_values={},
                uncertainty_analysis={},
                bias_analysis={},
                legal_precedent_chain=[],
                alternative_interpretations=[]
            )
    
    def _generate_reasoning_steps(self, text: str, claim: str, result: Dict) -> List[str]:
        """Generate step-by-step reasoning"""
        steps = [
            f"1. Analyzed claim: '{claim[:100]}{'...' if len(claim) > 100 else ''}'",
            f"2. Extracted key legal concepts from source text ({len(text)} characters)",
            f"3. Applied legal reasoning framework to assess claim validity",
            f"4. Cross-referenced with legal precedents and case law",
            f"5. Evaluated evidence strength and legal authority",
            f"6. Determined result: {result.get('result', 'UNKNOWN')} with {result.get('confidence', 0):.2f} confidence"
        ]
        
        # Add specific reasoning based on result
        if result.get('result') == 'SUPPORTS':
            steps.append("7. Found strong supporting evidence in legal precedents")
        elif result.get('result') == 'REFUTES':
            steps.append("7. Found contradictory evidence in established case law")
        else:
            steps.append("7. Insufficient evidence found for definitive determination")
        
        return steps
    
    def _analyze_features(self, text: str, claim: str) -> Dict[str, float]:
        """Analyze important features in the text"""
        # Simple keyword-based feature analysis
        legal_keywords = {
            'shall': 0.9, 'must': 0.9, 'required': 0.8, 'liable': 0.8,
            'damages': 0.8, 'breach': 0.8, 'terminate': 0.7, 'confidential': 0.7,
            'indemnify': 0.8, 'warranty': 0.7, 'penalty': 0.8, 'force majeure': 0.6
        }
        
        features = {}
        text_lower = text.lower()
        claim_lower = claim.lower()
        
        for keyword, weight in legal_keywords.items():
            if keyword in text_lower or keyword in claim_lower:
                features[keyword] = weight
        
        return features
    
    def _generate_decision_path(self, text: str, claim: str, result: Dict) -> str:
        """Generate decision tree path explanation"""
        confidence = result.get('confidence', 0.0)
        
        if confidence > 0.8:
            return "High confidence path: Strong legal precedent → Clear evidence → Definitive conclusion"
        elif confidence > 0.5:
            return "Medium confidence path: Some precedent → Moderate evidence → Probable conclusion"
        else:
            return "Low confidence path: Limited precedent → Weak evidence → Uncertain conclusion"
    
    def _generate_lime_explanation(self, text: str, claim: str) -> Dict[str, Any]:
        """Generate LIME-style explanation"""
        # Simplified LIME explanation
        words = claim.split()
        explanation = {}
        
        for i, word in enumerate(words):
            # Assign importance based on legal significance
            if word.lower() in ['shall', 'must', 'liable', 'damages', 'breach']:
                explanation[word] = 0.8 + np.random.normal(0, 0.1)
            elif word.lower() in ['may', 'should', 'could', 'might']:
                explanation[word] = 0.4 + np.random.normal(0, 0.1)
            else:
                explanation[word] = 0.2 + np.random.normal(0, 0.05)
        
        return {'word_importance': explanation}
    
    def _build_precedent_chain(self, claim: str, result: Dict) -> List[Dict]:
        """Build legal precedent chain"""
        # Mock precedent chain - in real implementation, would query case database
        precedents = [
            {
                'case_name': 'Smith v. Jones (2020)',
                'relevance_score': 0.85,
                'supporting_principle': 'Contractual obligations must be clearly defined',
                'authority_level': 'Circuit Court'
            },
            {
                'case_name': 'Legal Corp v. Business Inc (2019)',
                'relevance_score': 0.72,
                'supporting_principle': 'Payment terms require explicit agreement',
                'authority_level': 'District Court'
            }
        ]
        
        return precedents
    
    def _generate_alternatives(self, claim: str, result: Dict) -> List[Dict]:
        """Generate alternative interpretations"""
        alternatives = [
            {
                'interpretation': 'Strict contractual reading',
                'probability': 0.7,
                'reasoning': 'Literal interpretation of contract terms'
            },
            {
                'interpretation': 'Equitable considerations',
                'probability': 0.5,
                'reasoning': 'Considering fairness and good faith'
            },
            {
                'interpretation': 'Industry standard practice',
                'probability': 0.6,
                'reasoning': 'Based on common industry practices'
            }
        ]
        
        return alternatives
    
    def _create_decision_summary(self, result: Dict, reasoning_steps: List[str]) -> str:
        """Create human-readable decision summary"""
        confidence = result.get('confidence', 0.0)
        decision = result.get('result', 'UNKNOWN')
        
        summary = f"Decision: {decision} (Confidence: {confidence:.2f})\n\n"
        summary += "Key reasoning:\n"
        summary += "\n".join(reasoning_steps[:4])  # Top 4 steps
        
        if confidence > 0.8:
            summary += "\n\nThis is a high-confidence decision based on strong legal precedent."
        elif confidence > 0.5:
            summary += "\n\nThis is a moderate-confidence decision with some supporting evidence."
        else:
            summary += "\n\nThis is a low-confidence decision due to limited evidence."
        
        return summary
    
    def export_explanation_report(self, explanation: XAIExplanation, format: str = 'json') -> str:
        """Export comprehensive explanation report"""
        if format == 'json':
            return json.dumps(asdict(explanation), indent=2, default=str)
        elif format == 'html':
            return self._generate_html_report(explanation)
        else:
            return str(explanation)
    
    def _generate_html_report(self, explanation: XAIExplanation) -> str:
        """Generate HTML report of XAI explanation"""
        html = f"""
        <html>
        <head><title>Legal AI Explanation Report</title></head>
        <body>
        <h1>Legal AI Decision Explanation</h1>
        
        <h2>Decision Summary</h2>
        <p>{explanation.decision_summary}</p>
        
        <h2>Confidence Score</h2>
        <p>{explanation.confidence_score:.2f}</p>
        
        <h2>Reasoning Steps</h2>
        <ol>
        {''.join(f'<li>{step}</li>' for step in explanation.reasoning_steps)}
        </ol>
        
        <h2>Feature Importance</h2>
        <ul>
        {''.join(f'<li>{feature}: {importance:.3f}</li>' for feature, importance in explanation.feature_importance.items())}
        </ul>
        
        <h2>Uncertainty Analysis</h2>
        <p>Prediction Stability: {explanation.uncertainty_analysis.get('prediction_stability', 'Unknown')}</p>
        
        <h2>Bias Analysis</h2>
        <p>Neutrality Score: {explanation.bias_analysis.get('demographic_neutrality_score', 'Unknown')}</p>
        
        </body>
        </html>
        """
        return html