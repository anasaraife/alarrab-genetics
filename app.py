# ==============================================================================
#  العرّاب للجينات - الإصدار 14.0 (المحرك المتكامل المحسن)
#  - يدمج الواجهة العصرية مع العقل متعدد النماذج والحاسبة المدمجة
# ==============================================================================

import streamlit as st
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import PyPDF2
import os
import tempfile
import requests
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import hashlib
from datetime import datetime
from itertools import product
import collections
import pandas as pd
import io
import re

# --- التحقق من توفر المكتبات الاختيارية ---
try:
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

# -------------------------------------------------
#  1. إعدادات الصفحة والتصميم
# -------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="العرّاب للجينات V14.0",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# --- CSS متقدم للواجهة العصرية ---
st.markdown("""
<style>
    /* إخفاء العناصر الافتراضية */
    .stDeployButton, #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding: 0 !important; }
    
    /* الخلفية والتخطيط العام */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* حاوية المحادثة الرئيسية */
    .chat-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 0;
        margin: 20px auto;
        max-width: 1000px;
        height: 95vh;
        display: flex;
        flex-direction: column;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* شريط العنوان */
    .header-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-radius: 20px 20px 0 0;
        flex-shrink: 0;
    }
    
    .header-title { font-size: 24px; font-weight: bold; margin: 0; display: flex; align-items: center; gap: 15px; }
    .status-indicator { width: 10px; height: 10px; background: #00ff88; border-radius: 50%; animation: pulse 2s infinite; box-shadow: 0 0 8px #00ff88; }
    
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    
    /* منطقة المحادثة */
    .chat-area { flex-grow: 1; overflow-y: auto; padding: 20px 30px; }
    
    /* رسائل المحادثة */
    .message { margin-bottom: 20px; animation: slideIn 0.3s ease-out; }
    @keyframes slideIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
    
    .user-message { display: flex; justify-content: flex-end; }
    .user-bubble { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 18px; border-radius: 20px 20px 5px 20px; max-width: 80%; word-wrap: break-word; }
    
    .assistant-message { display: flex; align-items: flex-start; gap: 15px; }
    .avatar { width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 20px; color: white; flex-shrink: 0; }
    .assistant-bubble { background: #f1f3f5; border: 1px solid #e9ecef; padding: 15px 20px; border-radius: 20px 20px 20px 5px; max-width: calc(100% - 55px); word-wrap: break-word; }
    
    /* منطقة الإدخال */
    .input-area { padding: 15px 25px; background: #ffffff; border-radius: 0 0 20px 20px; border-top: 1px solid #e9ecef; flex-shrink: 0; }
    
    /* أزرار التشغيل السريع */
    .quick-actions { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
    .quick-btn { background: #e9ecef; border: none; color: #495057; padding: 6px 14px; border-radius: 15px; cursor: pointer; transition: all 0.2s ease; font-size: 13px; }
    .quick-btn:hover { background: #dee2e6; transform: translateY(-1px); }
    
    /* الحاسبة الوراثية المدمجة */
    .genetics-calculator { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 15px; padding: 20px; margin-top: 15px; }
    .calc-header { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #495057; }
    
    /* مصادر المعلومات */
    .sources-section { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 15px; margin-top: 15px; }
    .source-item { margin-bottom: 8px; padding: 8px; background: white; border-radius: 5px; border-left: 3px solid #667eea; }
    
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
#  2. قواعد البيانات والمدراء
# -------------------------------------------------

# --- قاعدة بيانات الجينات (للحاسبة) ---
GENE_DATA = {
    'B': {'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked', 'emoji': '🎨', 'alleles': {'BA': 'آش ريد', '+': 'أزرق/أسود', 'b': 'بني'}, 'dominance': ['BA', '+', 'b']},
    'd': {'display_name_ar': "التخفيف", 'type_en': 'sex-linked', 'emoji': '💧', 'alleles': {'+': 'عادي', 'd': 'مخفف'}, 'dominance': ['+', 'd']},
    'e': {'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal', 'emoji': '🔴', 'alleles': {'+': 'عادي', 'e': 'أحمر متنحي'}, 'dominance': ['+', 'e']},
    'C': {'display_name_ar': "النمط", 'type_en': 'autosomal', 'emoji': '📐', 'alleles': {'CT': 'نمط تي', 'C': 'تشيكر', '+': 'بار', 'c': 'بدون بار'}, 'dominance': ['CT', 'C', '+', 'c']},
    'S': {'display_name_ar': "الانتشار", 'type_en': 'autosomal', 'emoji': '🌊', 'alleles': {'S': 'منتشر', '+': 'عادي'}, 'dominance': ['S', '+']}
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {g: {n: s for s, n in d['alleles'].items()} for g, d in GENE_DATA.items()}

# --- روابط الكتب والمصادر ---
BOOK_LINKS = {
    "كتاب وراثة الحمام الأساسي": "https://drive.google.com/file/d/1ABC123/view",
    "دليل الألوان الوراثية": "https://drive.google.com/file/d/2DEF456/view", 
    "أسرار التربية المتقدمة": "https://drive.google.com/file/d/3GHI789/view"
}

# --- مدير نماذج الذكاء الاصطناعي المحسن ---
class AIModelManager:
    def __init__(self):
        self.models = {
            "gemini": {
                "name": "Google Gemini", 
                "available": self._check_secret("GEMINI_API_KEY"), 
                "priority": 1,
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
            },
            "deepseek": {
                "name": "DeepSeek", 
                "available": self._check_secret("DEEPSEEK_API_KEY"), 
                "priority": 2,
                "endpoint": "https://api.deepseek.com/v1/chat/completions"
            },
            "openai": {
                "name": "OpenAI GPT", 
                "available": self._check_secret("OPENAI_API_KEY"), 
                "priority": 3,
                "endpoint": "https://api.openai.com/v1/chat/completions"
            }
        }
    
    def _check_secret(self, key: str) -> bool:
        try: 
            return st.secrets.get(key) is not None and st.secrets[key] != ""
        except Exception: 
            return False
    
    def get_available_models(self) -> List[str]:
        available = [model for model, config in self.models.items() if config["available"]]
        return sorted(available, key=lambda x: self.models[x]["priority"])
    
    def get_model_info(self, model_key: str) -> Dict:
        return self.models.get(model_key, {})

# --- مدير المعرفة المحسن ---
class KnowledgeManager:
    def __init__(self, embedder=None):
        self.embedder = embedder
        self.db_path = os.path.join(tempfile.gettempdir(), "genetics_knowledge_v14.db")
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """إنشاء قاعدة البيانات وجداولها"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # إنشاء جدول المعرفة
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB
                )
            """)
            
            # إنشاء جدول الفهرسة للبحث السريع
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id INTEGER,
                    keywords TEXT,
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge (id)
                )
            """)
            
            self.conn.commit()
        except Exception as e:
            st.error(f"خطأ في إنشاء قاعدة البيانات: {e}")
    
    def add_content(self, source: str, content: str) -> bool:
        """إضافة محتوى جديد إلى قاعدة المعرفة"""
        try:
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            cursor = self.conn.cursor()
            
            # فحص وجود المحتوى
            cursor.execute("SELECT id FROM knowledge WHERE content_hash = ?", (content_hash,))
            if cursor.fetchone():
                return False  # المحتوى موجود مسبقاً
            
            # إنشاء embedding إذا كان متاحاً
            embedding_blob = None
            if self.embedder:
                try:
                    embedding = self.embedder.encode([content])[0]
                    embedding_blob = embedding.tobytes()
                except Exception as e:
                    st.warning(f"تعذر إنشاء embedding: {e}")
            
            # إدراج المحتوى
            cursor.execute("""
                INSERT INTO knowledge (source, content, content_hash, embedding)
                VALUES (?, ?, ?, ?)
            """, (source, content, content_hash, embedding_blob))
            
            knowledge_id = cursor.lastrowid
            
            # إنشاء فهرس الكلمات المفتاحية
            keywords = self._extract_keywords(content)
            cursor.execute("""
                INSERT INTO search_index (knowledge_id, keywords)
                VALUES (?, ?)
            """, (knowledge_id, keywords))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            st.error(f"خطأ في إضافة المحتوى: {e}")
            return False
    
    def _extract_keywords(self, content: str) -> str:
        """استخراج الكلمات المفتاحية من المحتوى"""
        # إزالة علامات الترقيم والرموز
        clean_content = re.sub(r'[^\w\s]', ' ', content)
        words = clean_content.split()
        
        # فلترة الكلمات القصيرة والشائعة
        stop_words = {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return ' '.join(keywords[:50])  # أخذ أول 50 كلمة مفتاحية
    
    def search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """البحث في قاعدة المعرفة"""
        try:
            cursor = self.conn.cursor()
            results = []
            
            # البحث النصي أولاً
            text_results = self._text_search(query, limit)
            results.extend(text_results)
            
            # البحث المتجه إذا كان متاحاً
            if self.embedder and len(results) < limit:
                vector_results = self._vector_search(query, limit - len(results))
                results.extend(vector_results)
            
            # إزالة التكرارات وترتيب النتائج
            seen_ids = set()
            unique_results = []
            for result in results:
                if result['id'] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result['id'])
            
            return unique_results[:limit]
            
        except Exception as e:
            st.error(f"خطأ في البحث: {e}")
            return []
    
    def _text_search(self, query: str, limit: int) -> List[Dict]:
        """البحث النصي التقليدي"""
        try:
            cursor = self.conn.cursor()
            
            # تقسيم الاستعلام إلى كلمات
            query_words = query.split()
            
            # بناء استعلام SQL للبحث في المحتوى والكلمات المفتاحية
            placeholders = ' OR '.join(['content LIKE ?' for _ in query_words])
            sql = f"""
                SELECT DISTINCT k.id, k.source, k.content, 
                       (CASE 
                        WHEN {placeholders} THEN 1 
                        ELSE 0 
                        END) as relevance_score
                FROM knowledge k
                LEFT JOIN search_index si ON k.id = si.knowledge_id
                WHERE {placeholders}
                ORDER BY relevance_score DESC
                LIMIT ?
            """
            
            params = [f'%{word}%' for word in query_words] * 2 + [limit]
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'source': row[1],
                    'content': row[2][:500] + '...' if len(row[2]) > 500 else row[2],
                    'full_content': row[2],
                    'score': row[3]
                })
            
            return results
            
        except Exception as e:
            st.warning(f"خطأ في البحث النصي: {e}")
            return []
    
    def _vector_search(self, query: str, limit: int) -> List[Dict]:
        """البحث المتجه باستخدام التشابه الدلالي"""
        try:
            cursor = self.conn.cursor()
            
            # إنشاء embedding للاستعلام
            query_embedding = self.embedder.encode([query])[0]
            
            # جلب جميع embeddings من قاعدة البيانات
            cursor.execute("SELECT id, source, content, embedding FROM knowledge WHERE embedding IS NOT NULL")
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            # حساب التشابه
            similarities = []
            for row in rows:
                try:
                    stored_embedding = np.frombuffer(row[3], dtype=np.float32)
                    similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
                    
                    if similarity > 0.3:  # عتبة التشابه
                        similarities.append({
                            'id': row[0],
                            'source': row[1],
                            'content': row[2][:500] + '...' if len(row[2]) > 500 else row[2],
                            'full_content': row[2],
                            'score': float(similarity)
                        })
                except Exception as e:
                    continue
            
            # ترتيب النتائج حسب التشابه
            similarities.sort(key=lambda x: x['score'], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            st.warning(f"خطأ في البحث المتجه: {e}")
            return []
    
    def get_knowledge_stats(self) -> Dict:
        """إحصائيات قاعدة المعرفة"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            total_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT source) FROM knowledge")
            total_sources = cursor.fetchone()[0]
            
            return {
                'total_documents': total_docs,
                'total_sources': total_sources,
                'vector_search_enabled': self.embedder is not None
            }
        except Exception:
            return {'total_documents': 0, 'total_sources': 0, 'vector_search_enabled': False}

# --- الحاسبة الوراثية المحسنة ---
class AdvancedGeneticCalculator:
    def describe_phenotype(self, gt_dict: Dict) -> Tuple[str, str]:
        """وصف النمط الظاهري من النمط الوراثي"""
        phenotypes = {g: "" for g in GENE_ORDER}
        
        for gene, gt_part in gt_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dom_allele in GENE_DATA[gene]['dominance']:
                if dom_allele in alleles:
                    phenotypes[gene] = GENE_DATA[gene]['alleles'][dom_allele]
                    break
        
        # معالجة الأحمر المتنحي
        if 'e//e' in gt_dict.get('e', ''):
            phenotypes['B'] = 'أحمر متنحي'
            phenotypes['C'] = ''
        
        # معالجة الانتشار
        if 'S' in gt_dict.get('S', ''):
            if 'e//e' not in gt_dict.get('e', ''):
                phenotypes['C'] = 'منتشر'
        
        # تحديد الجنس
        sex = "أنثى" if any('•' in gt_dict.get(g, '') for g, d in GENE_DATA.items() if d['type_en'] == 'sex-linked') else "ذكر"
        
        # بناء الوصف
        desc_parts = [
            phenotypes.get('B'),
            'مخفف' if phenotypes.get('d') == 'مخفف' else None,
            phenotypes.get('C')
        ]
        
        phenotype_desc = f"{sex} {' '.join(filter(None, desc_parts))}"
        genotype_str = " | ".join([gt_dict[g].strip() for g in GENE_ORDER])
        
        return phenotype_desc, genotype_str

    def calculate(self, parent_inputs: Dict) -> Dict:
        """حساب نتائج التزاوج"""
        try:
            parent_gts = {}
            
            # بناء الأنماط الوراثية للوالدين
            for parent in ['male', 'female']:
                gt_parts = []
                for gene in GENE_ORDER:
                    info = GENE_DATA[gene]
                    vis = parent_inputs[parent].get(f'{gene}_visible')
                    hid = parent_inputs[parent].get(f'{gene}_hidden', vis)
                    
                    # تحويل الأسماء إلى رموز
                    vis_sym = NAME_TO_SYMBOL_MAP[gene].get(vis, info['dominance'][0])
                    hid_sym = NAME_TO_SYMBOL_MAP[gene].get(hid, vis_sym)
                    
                    if info['type_en'] == 'sex-linked' and parent == 'female':
                        gt_parts.append(f"•//{vis_sym}")
                    else:
                        # ترتيب الأليلات حسب الهيمنة
                        alleles = sorted([vis_sym, hid_sym], key=lambda x: info['dominance'].index(x))
                        gt_parts.append(f"{alleles[0]}//{alleles[1]}")
                
                parent_gts[parent] = gt_parts
            
            # إنتاج الأمشاج
            def get_gametes(gt_parts, is_female):
                parts_for_prod = []
                for i, part in enumerate(gt_parts):
                    gene = GENE_ORDER[i]
                    if GENE_DATA[gene]['type_en'] == 'sex-linked' and is_female:
                        parts_for_prod.append([part.replace('•//','').strip()])
                    else:
                        parts_for_prod.append(part.split('//'))
                return list(product(*parts_for_prod))

            male_gametes = get_gametes(parent_gts['male'], False)
            female_gametes = get_gametes(parent_gts['female'], True)
            
            # حساب النسل
            offspring = collections.Counter()
            for m_g in male_gametes:
                for f_g in female_gametes:
                    son_gt, daughter_gt = {}, {}
                    
                    for i, gene in enumerate(GENE_ORDER):
                        alleles = sorted([m_g[i], f_g[i]], key=lambda x: GENE_DATA[gene]['dominance'].index(x))
                        
                        if GENE_DATA[gene]['type_en'] == 'sex-linked':
                            son_gt[gene] = f"{alleles[0]}//{alleles[1]}"
                            daughter_gt[gene] = f"•//{m_g[i]}"
                        else:
                            gt = f"{alleles[0]}//{alleles[1]}"
                            son_gt[gene] = gt
                            daughter_gt[gene] = gt
                    
                    offspring[self.describe_phenotype(son_gt)] += 1
                    offspring[self.describe_phenotype(daughter_gt)] += 1
            
            total = sum(offspring.values())
            
            return {
                'results': offspring,
                'total': total,
                'parent_genotypes': parent_gts,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f"خطأ في الحساب: {str(e)}",
                'success': False
            }

# -------------------------------------------------
#  3. نظام الرد الذكي المحسن
# -------------------------------------------------
class IntelligentResponder:
    def __init__(self, ai_manager, knowledge_manager):
        self.ai_manager = ai_manager
        self.knowledge_manager = knowledge_manager
        self.available_models = ai_manager.get_available_models()
        
        # قاموس المصطلحات الوراثية
        self.genetics_terms = {
            'جين': 'gene', 'أليل': 'allele', 'نمط وراثي': 'genotype',
            'نمط ظاهري': 'phenotype', 'هيمنة': 'dominance', 'تنحي': 'recessive',
            'مرتبط بالجنس': 'sex-linked', 'جسمي': 'autosomal'
        }

    def understand_intent(self, query: str) -> Dict:
        """فهم نية المستخدم من الاستعلام"""
        query_lower = query.lower()
        
        # كلمات مفتاحية للحساب
        calc_keywords = ['احسب', 'حساب', 'نتائج', 'تزاوج', 'تربية', 'نسل']
        
        # كلمات مفتاحية للألوان
        color_keywords = ['لون', 'ألوان', 'أحمر', 'أزرق', 'بني', 'آش ريد']
        
        # كلمات مفتاحية للوراثة
        genetics_keywords = ['وراثة', 'جين', 'جينات', 'أليل', 'نمط']
        
        intent = {'type': 'general', 'confidence': 0.5, 'keywords': []}
        
        if any(word in query_lower for word in calc_keywords):
            intent = {'type': 'calculation', 'confidence': 0.9, 'keywords': calc_keywords}
        elif any(word in query_lower for word in color_keywords):
            intent = {'type': 'colors', 'confidence': 0.8, 'keywords': color_keywords}
        elif any(word in query_lower for word in genetics_keywords):
            intent = {'type': 'genetics', 'confidence': 0.8, 'keywords': genetics_keywords}
        
        return intent

    def generate_response(self, query: str) -> Dict:
        """توليد الرد الذكي"""
        try:
            # فهم النية
            intent = self.understand_intent(query)
            
            # إذا كان الطلب للحساب، عرض الحاسبة
            if intent['type'] == 'calculation':
                return {
                    "answer": "🧮 بالتأكيد! تفضل باستخدام الحاسبة الوراثية أدناه لحساب نتائج التزاوج.",
                    "show_calculator": True,
                    "sources": [],
                    "intent": intent
                }
            
            # البحث في قاعدة المعرفة
            context_docs = self.knowledge_manager.search_content(query, limit=3)
            
            # محاولة الحصول على رد من النماذج المتاحة
            for model_key in self.available_models:
                try:
                    answer = self._get_model_response(model_key, query, context_docs, intent)
                    
                    if answer and "خطأ" not in answer and len(answer.strip()) > 10:
                        return {
                            "answer": answer,
                            "show_calculator": False,
                            "sources": context_docs,
                            "model_used": model_key,
                            "intent": intent
                        }
                        
                except Exception as e:
                    st.warning(f"خطأ في نموذج {model_key}: {e}")
                    continue
            
            # رد افتراضي إذا فشل جميع النماذج
            fallback_answer = self._generate_fallback_response(query, context_docs, intent)
            return {
                "answer": fallback_answer,
                "show_calculator": False,
                "sources": context_docs,
                "model_used": "fallback",
                "intent": intent
            }
            
        except Exception as e:
            return {
                "answer": f"عذراً، حدث خطأ في معالجة استفسارك: {str(e)}",
                "show_calculator": False,
                "sources": [],
                "intent": {'type': 'error'}
            }

    def _get_model_response(self, model_key: str, query: str, context_docs: List[Dict], intent: Dict) -> str:
        """الحصول على رد من نموذج ذكي محدد"""
        if model_key == "gemini":
            return self._get_gemini_response(query, context_docs, intent)
        elif model_key == "deepseek":
            return self._get_deepseek_response(query, context_docs, intent)
        elif model_key == "openai":
            return self._get_openai_response(query, context_docs, intent)
        else:
            return "نموذج غير مدعوم"

    def _build_context_prompt(self, query: str, context_docs: List[Dict], intent: Dict) -> str:
        """بناء prompt السياق للنماذج الذكية"""
        context_text = ""
        if context_docs:
            context_text = "\n\n".join([
                f"المصدر: {doc['source']}\nالمحتوى: {doc['full_content'][:800]}..."
                for doc in context_docs
            ])
        
        intent_instruction = ""
        if intent['type'] == 'colors':
            intent_instruction = "ركز على شرح الألوان الوراثية وأنماطها."
        elif intent['type'] == 'genetics':
            intent_instruction = "اشرح المفاهيم الوراثية بشكل مفصل ومبسط."
        
        prompt = f"""أنت خبير في وراثة الحمام تجيب باللغة العربية فقط.

السياق المتاح:
{context_text if context_text else "لا يوجد سياق محدد"}

تعليمات خاصة: {intent_instruction}

سؤال المستخدم: {query}

يرجى الإجابة بشكل دقيق ومفيد باللغة العربية، واستخدم المعلومات من السياق إذا كانت متوفرة. إذا لم تكن متأكداً من المعلومة، اذكر ذلك بوضوح."""

        return prompt

    def _get_gemini_response(self, query: str, context_docs: List[Dict], intent: Dict) -> str:
        """الحصول على رد من Gemini"""
        try:
            API_KEY = st.secrets["GEMINI_API_KEY"]
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            
            prompt = self._build_context_prompt(query, context_docs, intent)
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
            
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                return "خطأ في استجابة Gemini"
                
        except requests.exceptions.Timeout:
            return "خطأ: انتهت مهلة الاتصال مع Gemini"
        except requests.exceptions.RequestException as e:
            return f"خطأ في الاتصال مع Gemini: {e}"
        except Exception as e:
            return f"خطأ في معالجة رد Gemini: {e}"

    def _get_deepseek_response(self, query: str, context_docs: List[Dict], intent: Dict) -> str:
        """الحصول على رد من DeepSeek"""
        try:
            API_KEY = st.secrets["DEEPSEEK_API_KEY"]
            API_URL = "https://api.deepseek.com/v1/chat/completions"
            
            prompt = self._build_context_prompt(query, context_docs, intent)
            
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content'].strip()
            else:
                return "خطأ في استجابة DeepSeek"
                
        except requests.exceptions.Timeout:
            return "خطأ: انتهت مهلة الاتصال مع DeepSeek"
        except Exception as e:
            return f"خطأ في DeepSeek: {e}"

    def _get_openai_response(self, query: str, context_docs: List[Dict], intent: Dict) -> str:
        """الحصول على رد من OpenAI"""
        try:
            API_KEY = st.secrets["OPENAI_API_KEY"]
            API_URL = "https://api.openai.com/v1/chat/completions"
            
            prompt = self._build_context_prompt(query, context_docs, intent)
            
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            }
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content'].strip()
            else:
                return "خطأ في استجابة OpenAI"
                
        except Exception as e:
            return f"خطأ في OpenAI: {e}"

    def _generate_fallback_response(self, query: str, context_docs: List[Dict], intent: Dict) -> str:
        """توليد رد احتياطي بدون نماذج ذكية"""
        if context_docs:
            # استخدام المحتوى المتاح لتكوين رد أساسي
            relevant_content = context_docs[0]['content']
            return f"بناءً على المعلومات المتاحة:\n\n{relevant_content}\n\nللمزيد من التفاصيل، يمكنك استخدام الحاسبة الوراثية أو طرح سؤال أكثر تحديداً."
        
        # ردود افتراضية حسب النية
        if intent['type'] == 'colors':
            return """🎨 ألوان الحمام الوراثية:

الألوان الأساسية:
• الأزرق/الأسود (+) - اللون الأساسي الطبيعي
• البني (b) - متنحي للأزرق
• الآش ريد (BA) - مهيمن على الأزرق والبني

الأحمر المتنحي (e):
• يحول أي لون أساسي إلى أحمر عند وجوده في حالة متنحية (e/e)

التخفيف (d):
• يخفف كثافة اللون الأساسي
• مرتبط بالجنس مثل اللون الأساسي

لحساب نتائج التزاوج بدقة، استخدم الحاسبة الوراثية المدمجة."""
        
        elif intent['type'] == 'genetics':
            return """🧬 أساسيات الوراثة في الحمام:

المفاهيم الأساسية:
• النمط الوراثي (Genotype): التركيب الجيني الفعلي
• النمط الظاهري (Phenotype): المظهر الخارجي المرئي
• الهيمنة: قدرة أليل على إخفاء تأثير أليل آخر
• التنحي: الأليل الذي لا يظهر إلا في حالة التماثل

أنواع الوراثة:
• مرتبط بالجنس: الجينات على الكروموسوم الجنسي
• جسمي: الجينات على الكروموسومات العادية

استخدم الحاسبة لتطبيق هذه المفاهيم عملياً."""
        
        else:
            return """مرحباً بك في العرّاب للجينات! 

يمكنني مساعدتك في:
🧮 حساب نتائج التزاوج الوراثي
🎨 شرح ألوان الحمام وأنماطها  
🧬 توضيح المفاهيم الوراثية
📚 الإجابة على أسئلة التربية

حالياً، النماذج الذكية غير متاحة، لكن يمكنك استخدام الحاسبة الوراثية أو طرح سؤال أكثر تحديداً."""

# -------------------------------------------------
#  4. تحميل الموارد وبناء قاعدة المعرفة
# -------------------------------------------------
@st.cache_resource
def load_resources():
    """تحميل موارد النظام"""
    resources = {"embedder": None, "knowledge_manager": None}
    
    # تحميل نموذج التضمين إذا كان متاحاً
    if VECTOR_SEARCH_AVAILABLE:
        try:
            with st.spinner("تحميل نموذج البحث الدلالي..."):
                resources["embedder"] = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        except Exception as e:
            st.warning(f"تعذر تحميل نموذج البحث الدلالي: {e}")
    
    # إنشاء مدير المعرفة
    resources["knowledge_manager"] = KnowledgeManager(resources["embedder"])
    
    return resources

def load_sample_knowledge(knowledge_manager):
    """تحميل معرفة تجريبية"""
    sample_content = [
        {
            "source": "دليل الألوان الأساسية",
            "content": """الألوان الأساسية في الحمام:

الأزرق/الأسود (+): هو اللون الأساسي الطبيعي، يظهر كأزرق مع أشرطة سوداء أو كأسود تام حسب النمط.

البني (b): لون متنحي للأزرق، يحول الأزرق إلى بني والأسود إلى شوكولاتي.

الآش ريد (BA): لون مهيمن على جميع الألوان الأخرى، يعطي لوناً أحمر مائل للرمادي.

الأحمر المتنحي (e): عندما يكون في حالة متنحية (e/e) يحول أي لون إلى أحمر صافي.

التخفيف (d): يخفف كثافة اللون، فيحول الأزرق إلى فضي والأسود إلى دن والأحمر إلى أصفر."""
        },
        {
            "source": "قواعد الوراثة الأساسية", 
            "content": """قوانين مندل في وراثة الحمام:

قانون الهيمنة: الأليل المهيمن يخفي تأثير الأليل المتنحي في الأفراد متغايرة الأقران.

قانون الفصل: أليلات الجين الواحد تنفصل أثناء تكوين الأمشاج، فكل مشيج يحمل أليل واحد فقط.

قانون التشكيل المستقل: جينات مختلفة تورث بشكل مستقل عن بعضها البعض.

الوراثة المرتبطة بالجنس: في الحمام، الإناث لديها كروموسوم جنسي واحد (ZW) بينما الذكور لديهم اثنان (ZZ)، لذلك الإناث تظهر صفات الأليل الواحد مباشرة."""
        },
        {
            "source": "أنماط الريش والأشرطة",
            "content": """أنماط الريش في الحمام:

البار (+): النمط الأساسي، يظهر شريطين أسودين على الجناح.

التشيكر (C): نمط مهيمن على البار، يظهر نقط أو رقع صغيرة منتشرة على الريش.

نمط التي (CT): الأكثر هيمنة، يعطي لوناً موحداً بدون أشرطة أو نقط.

بدون بار (c): نمط متنحي، لا يظهر أي أشرطة على الجناح.

الانتشار (S): يوزع لون الشريط أو النقط على كامل الريشة، مما يعطي مظهراً أغمق وأكثر كثافة."""
        }
    ]
    
    # إضافة المحتوى التجريبي
    for item in sample_content:
        knowledge_manager.add_content(item["source"], item["content"])

# -------------------------------------------------
#  5. الواجهة الرئيسية
# -------------------------------------------------
def initialize_session_state():
    """تهيئة حالة الجلسة"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_loaded" not in st.session_state:
        st.session_state.knowledge_loaded = False

def render_sources_section(sources: List[Dict]):
    """عرض مصادر المعلومات"""
    if not sources:
        return
    
    st.markdown("""
    <div class="sources-section">
        <strong>🔍 المصادر المستخدمة:</strong>
    """, unsafe_allow_html=True)
    
    for i, source in enumerate(sources):
        st.markdown(f"""
        <div class="source-item">
            <strong>{source['source']}</strong><br>
            <small>{source['content'][:150]}{'...' if len(source['content']) > 150 else ''}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_embedded_calculator():
    """عرض الحاسبة الوراثية المدمجة"""
    with st.container():
        st.markdown('<div class="genetics-calculator">', unsafe_allow_html=True)
        st.markdown('<div class="calc-header">🧮 الحاسبة الوراثية المدمجة</div>', unsafe_allow_html=True)
        
        parent_inputs = {'male': {}, 'female': {}}
        col1, col2 = st.columns(2)
        
        # إدخال بيانات الوالدين
        for parent, col in [('male', col1), ('female', col2)]:
            with col:
                st.markdown(f"#### {'♂️ الذكر' if parent == 'male' else '♀️ الأنثى'}")
                
                for gene, data in GENE_DATA.items():
                    choices = list(data['alleles'].values())
                    
                    # الأليل الظاهر
                    parent_inputs[parent][f'{gene}_visible'] = st.selectbox(
                        f"{data['emoji']} {data['display_name_ar']} (الظاهر):",
                        choices,
                        key=f"emb_{parent}_{gene}_vis"
                    )
                    
                    # الأليل المخفي (للذكور وللجينات الجسمية في الإناث)
                    if not (data['type_en'] == 'sex-linked' and parent == 'female'):
                        parent_inputs[parent][f'{gene}_hidden'] = st.selectbox(
                            f"{data['emoji']} {data['display_name_ar']} (المخفي):",
                            choices,
                            key=f"emb_{parent}_{gene}_hid",
                            index=choices.index(parent_inputs[parent][f'{gene}_visible'])
                        )
                    else:
                        parent_inputs[parent][f'{gene}_hidden'] = parent_inputs[parent][f'{gene}_visible']
        
        # زر الحساب
        if st.button("🚀 احسب النتائج", use_container_width=True, type="primary"):
            calculator = AdvancedGeneticCalculator()
            result_data = calculator.calculate(parent_inputs)
            
            if not result_data.get('success', False):
                st.error(result_data.get('error', 'خطأ غير معروف'))
            else:
                # عرض النتائج في جدول
                results_list = []
                for (phenotype, genotype), count in result_data['results'].items():
                    percentage = (count / result_data['total']) * 100
                    results_list.append({
                        'النمط الظاهري': phenotype,
                        'النمط الوراثي': genotype,
                        'العدد': count,
                        'النسبة %': f"{percentage:.1f}%"
                    })
                
                df = pd.DataFrame(results_list)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # إحصائيات إضافية
                with st.expander("📊 إحصائيات مفصلة"):
                    st.write(f"**إجمالي النسل المتوقع:** {result_data['total']}")
                    st.write(f"**عدد الأنماط المختلفة:** {len(result_data['results'])}")
                    
                    # الأنماط الوراثية للوالدين
                    st.write("**الأنماط الوراثية للوالدين:**")
                    for parent, genotype in result_data['parent_genotypes'].items():
                        parent_name = "الذكر" if parent == 'male' else "الأنثى"
                        st.write(f"- {parent_name}: {' | '.join(genotype)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def handle_user_message(prompt: str, responder: IntelligentResponder):
    """معالجة رسالة المستخدم"""
    # إضافة رسالة المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # توليد الرد
    with st.spinner("جاري التفكير..."):
        response_data = responder.generate_response(prompt)
    
    # إضافة رد المساعد
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_data["answer"],
        "show_calculator": response_data.get("show_calculator", False),
        "sources": response_data.get("sources", []),
        "model_used": response_data.get("model_used"),
        "intent": response_data.get("intent", {})
    })

def main():
    """الدالة الرئيسية"""
    initialize_session_state()
    
    # تحميل الموارد
    resources = load_resources()
    knowledge_manager = resources["knowledge_manager"]
    
    # تحميل المعرفة التجريبية إذا لم تكن محملة
    if not st.session_state.knowledge_loaded:
        with st.spinner("تحميل قاعدة المعرفة..."):
            load_sample_knowledge(knowledge_manager)
            st.session_state.knowledge_loaded = True
    
    # إنشاء المدراء
    ai_manager = AIModelManager()
    responder = IntelligentResponder(ai_manager, knowledge_manager)

    # الواجهة الرئيسية
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # شريط العنوان
    available_models = ai_manager.get_available_models()
    model_status = f"متاح ({len(available_models)} نماذج)" if available_models else "غير متاح"
    
    st.markdown(f'''
    <div class="header-bar">
        <div class="header-title">
            🧬 العرّاب للجينات V14.0
            <small style="font-size: 14px; opacity: 0.8;">محرك الذكاء: {model_status}</small>
        </div>
        <div style="font-size: 14px; display: flex; align-items: center; gap: 8px;">
            <div class="status-indicator"></div>
            نشط الآن
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # منطقة المحادثة
    chat_area = st.container()
    with chat_area:
        st.markdown('<div class="chat-area">', unsafe_allow_html=True)
        
        # رسالة الترحيب الأولى
        if not st.session_state.messages:
            welcome_msg = """مرحباً بك في العرّاب للجينات V14.0! 🧬

أنا مساعدك الذكي المتخصص في وراثة الحمام. يمكنني مساعدتك في:

🧮 **حساب نتائج التزاوج** - باستخدام الحاسبة الوراثية المتقدمة
🎨 **شرح الألوان والأنماط** - من الأساسيات إلى التفاصيل المعقدة  
🧬 **توضيح المفاهيم الوراثية** - بطريقة مبسطة ومفهومة
📚 **الإجابة على أسئلة التربية** - بناءً على أحدث المعلومات

كيف يمكنني مساعدتك اليوم؟"""
            
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        # عرض الرسائل
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'''
                <div class="message user-message">
                    <div class="user-bubble">{msg["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="message assistant-message">
                    <div class="avatar">🤖</div>
                    <div class="assistant-bubble">{msg["content"]}</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # عرض المصادر إن وجدت
                if msg.get("sources"):
                    render_sources_section(msg["sources"])
                
                # عرض الحاسبة إن طُلبت
                if msg.get("show_calculator"):
                    render_embedded_calculator()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # منطقة الإدخال
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # أزرار التشغيل السريع
    quick_actions = ["🧮 حساب وراثي", "🎨 شرح الألوان", "🧬 مفاهيم الوراثة", "💡 نصائح تربية"]
    cols = st.columns(len(quick_actions))
    
    for i, action in enumerate(quick_actions):
        if cols[i].button(action, use_container_width=True):
            handle_user_message(action, responder)
            st.rerun()

    # حقل الإدخال الرئيسي
    if prompt := st.chat_input("اكتب سؤالك هنا... 💬"):
        handle_user_message(prompt, responder)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # الشريط الجانبي للإحصائيات
    with st.sidebar:
        st.markdown("### 📊 إحصائيات النظام")
        
        # إحصائيات قاعدة المعرفة
        stats = knowledge_manager.get_knowledge_stats()
        st.metric("مستندات المعرفة", stats['total_documents'])
        st.metric("المصادر", stats['total_sources'])
        
        # حالة النماذج
        st.markdown("### 🤖 حالة النماذج")
        for model_key, config in ai_manager.models.items():
            status = "✅ متاح" if config['available'] else "❌ غير متاح"
            st.write(f"**{config['name']}:** {status}")
        
        # خيارات متقدمة
        with st.expander("⚙️ إعدادات متقدمة"):
            if st.button("🔄 إعادة تحميل المعرفة"):
                st.session_state.knowledge_loaded = False
                st.rerun()
            
            if st.button("🗑️ مسح المحادثة"):
                st.session_state.messages = []
                st.rerun()

if __name__ == "__main__":
    main()
