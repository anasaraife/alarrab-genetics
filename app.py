# ==============================================================================
#  العرّاب للجينات - الإصدار 6.0 (مع الوكيل الذكي التفاعلي)
# ==============================================================================

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import gdown
import PyPDF2
import os
import tempfile
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from datetime import datetime
import re

# -------------------------------------------------
#  1. إعدادات الصفحة
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الوكيل الذكي",
    page_icon="🕊️",
    layout="wide",
)

# قائمة روابط الكتب
BOOK_LINKS = [
    "https://drive.google.com/file/d/1CRwW78pd2RsKVd37elefz71RqwaCaute/view?usp=sharing",
    "https://drive.google.com/file/d/1894OOW1nEc3SkanLKKEzaXu_XhXYv8rF/view?usp=sharing",
    "https://drive.google.com/file/d/18pc9PptjfcjQfPyVCiaSq30RFs3ZjXF4/view?usp=sharing",
]

# قاعدة المعرفة الموسعة
GENETICS_KNOWLEDGE = {
    "genes": {
        "Blue/Black": {
            "symbol": "B+",
            "chromosome": "Z",
            "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون الأزرق/الأسود في الحمام. هذا الجين سائد ومرتبط بالجنس.",
            "phenotype": "لون أزرق رمادي أو أسود حسب وجود جينات أخرى مثل Spread",
            "breeding_info": "الذكور يحتاجون نسختين من الجين، الإناث نسخة واحدة فقط",
            "combinations": {
                "B+ + S": "أسود صلب",
                "B+ without S": "أزرق مع أنماط",
                "B+ + C": "أزرق شطرنج"
            }
        },
        "Ash-red": {
            "symbol": "BA",
            "chromosome": "Z", 
            "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون الأحمر الرمادي. سائد على Blue ومتنحي أمام Brown في بعض الحالات.",
            "phenotype": "لون أحمر رمادي مع تدرجات مختلفة من الوردي إلى الأحمر الداكن",
            "breeding_info": "ينتج ألوان جميلة عند التزاوج مع Blue",
            "combinations": {
                "BA + S": "أحمر صلب",
                "BA + C": "أحمر شطرنج",
                "BA + T": "أحمر مع خطوط"
            }
        },
        "Brown": {
            "symbol": "b",
            "chromosome": "Z",
            "inheritance": "Sex-linked", 
            "description": "الجين المسؤول عن اللون البني. متنحي أمام Blue و Ash-red.",
            "phenotype": "لون بني شوكولاتي عميق",
            "breeding_info": "نادر الظهور، يحتاج والدين حاملين للجين",
            "combinations": {
                "b + S": "بني صلب",
                "b + C": "بني شطرنج"
            }
        },
        "Checker": {
            "symbol": "C",
            "chromosome": "1",
            "inheritance": "Autosomal",
            "description": "نمط الشطرنج على الأجنحة. سائد جزئياً على T-pattern.",
            "phenotype": "نمط مربعات داكنة وفاتحة على الأجنحة يشبه رقعة الشطرنج",
            "breeding_info": "يظهر في كلا الجنسين بنفس الطريقة",
            "combinations": {
                "C + Blue": "شطرنج أزرق",
                "C + Ash-red": "شطرنج أحمر"
            }
        },
        "Spread": {
            "symbol": "S",
            "chromosome": "8",
            "inheritance": "Autosomal",
            "description": "انتشار اللون على كامل الطائر. يخفي جميع الأنماط الأخرى.",
            "phenotype": "لون موحد بدون أنماط أو خطوط على كامل الجسم",
            "breeding_info": "سائد، يحتاج نسخة واحدة فقط للظهور",
            "combinations": {
                "S + أي لون": "لون صلب بدون أنماط"
            }
        },
        "Red Bar": {
            "symbol": "T",
            "chromosome": "1",
            "inheritance": "Autosomal",
            "description": "الخطوط الحمراء على الأجنحة. البديل البري الأساسي.",
            "phenotype": "خطان أحمران عرضيان على كل جناح",
            "breeding_info": "النمط الأساسي الأكثر شيوعاً",
            "combinations": {
                "T + Blue": "أزرق مع خطوط",
                "T + Ash-red": "أحمر مع خطوط داكنة"
            }
        }
    },
    "breeding_patterns": {
        "sex_linked": "الجينات المرتبطة بالجنس تورث من الأب للبنات ومن الأم للأولاد",
        "autosomal": "الجينات الجسمية تورث بنفس الطريقة في كلا الجنسين",
        "dominance": "الجين السائد يظهر حتى لو كان موجود في نسخة واحدة فقط"
    },
    "common_questions": {
        "كيف أعرف جينات حمامتي": "يمكن معرفة الجينات من خلال اللون والنمط الظاهر، ولكن الاختبار الوراثي أدق",
        "ما أفضل تزاوج للحصول على ألوان جميلة": "تزاوج Ash-red مع Blue ينتج تنوع جميل في الألوان",
        "لماذا لا تظهر بعض الألوان في النسل": "قد تكون الجينات متنحية أو مخفية بواسطة جينات أخرى مثل Spread"
    }
}

# -------------------------------------------------
#  2. نظام الذكاء الاصطناعي للمحادثة
# -------------------------------------------------

class GeneticsAI:
    def __init__(self):
        self.knowledge = GENETICS_KNOWLEDGE
        self.conversation_history = []
        
    def analyze_query(self, query):
        """تحليل الاستعلام وتحديد نوع السؤال"""
        query_lower = query.lower()
        
        # أنواع الأسئلة
        question_types = {
            'gene_info': ['ما هو', 'اشرح', 'معلومات عن', 'تعريف'],
            'breeding': ['تزاوج', 'تربية', 'نسل', 'breeding', 'offspring'],
            'inheritance': ['وراثة', 'كيف يورث', 'inheritance', 'inherit'],
            'phenotype': ['لون', 'شكل', 'مظهر', 'نمط', 'color', 'pattern'],
            'comparison': ['مقارنة', 'فرق', 'أفضل', 'compare', 'difference']
        }
        
        detected_types = []
        for q_type, keywords in question_types.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_types.append(q_type)
        
        # البحث عن أسماء الجينات
        mentioned_genes = []
        for gene_name in self.knowledge['genes'].keys():
            if gene_name.lower() in query_lower or any(keyword in query_lower for keyword in [gene_name.split('/')[0].lower(), gene_name.split('/')[-1].lower() if '/' in gene_name else gene_name.lower()]):
                mentioned_genes.append(gene_name)
        
        return {
            'types': detected_types,
            'genes': mentioned_genes,
            'original_query': query
        }
    
    def generate_response(self, query):
        """توليد إجابة ذكية بناءً على تحليل الاستعلام"""
        analysis = self.analyze_query(query)
        
        # إضافة السؤال لتاريخ المحادثة
        self.conversation_history.append({'user': query, 'timestamp': datetime.now()})
        
        response = ""
        
        # الإجابة بناءً على نوع السؤال
        if 'gene_info' in analysis['types'] and analysis['genes']:
            response = self._explain_genes(analysis['genes'])
        elif 'breeding' in analysis['types']:
            response = self._breeding_advice(analysis['genes'])
        elif 'inheritance' in analysis['types']:
            response = self._inheritance_explanation(analysis['genes'])
        elif 'phenotype' in analysis['types']:
            response = self._phenotype_description(analysis['genes'])
        elif 'comparison' in analysis['types']:
            response = self._compare_genes(analysis['genes'])
        else:
            # الإجابة العامة أو البحث في الأسئلة الشائعة
            response = self._general_response(query)
        
        # إضافة الإجابة لتاريخ المحادثة
        self.conversation_history.append({'ai': response, 'timestamp': datetime.now()})
        
        return response
    
    def _explain_genes(self, genes):
        """شرح الجينات المحددة"""
        if not genes:
            return "لم أستطع تحديد الجين المقصود. يرجى ذكر اسم الجين بوضوح."
        
        explanations = []
        for gene in genes:
            if gene in self.knowledge['genes']:
                gene_info = self.knowledge['genes'][gene]
                explanation = f"""
🧬 **{gene} ({gene_info['symbol']})**

📍 **الموقع:** الكروموسوم {gene_info['chromosome']}
🔄 **نوع الوراثة:** {gene_info['inheritance']}

📝 **الوصف:**
{gene_info['description']}

🎨 **النمط الظاهري:**
{gene_info['phenotype']}

🐣 **معلومات التربية:**
{gene_info['breeding_info']}

🔀 **التركيبات الشائعة:**
"""
                for combo, result in gene_info['combinations'].items():
                    explanation += f"\n• {combo} → {result}"
                
                explanations.append(explanation)
        
        return "\n\n".join(explanations)
    
    def _breeding_advice(self, genes):
        """نصائح التربية والتزاوج"""
        advice = "💡 **نصائح التربية والتزاوج:**\n\n"
        
        if genes:
            for gene in genes:
                if gene in self.knowledge['genes']:
                    gene_info = self.knowledge['genes'][gene]
                    advice += f"**{gene}:** {gene_info['breeding_info']}\n\n"
        else:
            advice += """
📋 **نصائح عامة للتربية:**

1. **للحصول على ألوان متنوعة:** جرب تزاوج Ash-red مع Blue
2. **للألوان الصلبة:** استخدم الحمام الحامل لجين Spread
3. **للأنماط الجميلة:** تجنب استخدام Spread إذا كنت تريد رؤية الأنماط
4. **للجينات المرتبطة بالجنس:** الذكر يحدد لون الإناث، والأنثى تحدد لون الذكور
5. **للحصول على ألوان نادرة:** قد تحتاج عدة أجيال لظهور الجينات المتنحية

🔍 **تذكر:** الاختبار الوراثي هو الطريقة الأكثر دقة لمعرفة التركيب الوراثي الحقيقي.
"""
        
        return advice
    
    def _inheritance_explanation(self, genes):
        """شرح أنماط الوراثة"""
        explanation = "🔬 **أنماط الوراثة في الحمام:**\n\n"
        
        explanation += f"""
**الوراثة المرتبطة بالجنس (Sex-linked):**
{self.knowledge['breeding_patterns']['sex_linked']}
- أمثلة: Blue/Black, Ash-red, Brown

**الوراثة الجسمية (Autosomal):**
{self.knowledge['breeding_patterns']['autosomal']}
- أمثلة: Checker, Spread, Red Bar

**السيادة والتنحي:**
{self.knowledge['breeding_patterns']['dominance']}
"""
        
        if genes:
            explanation += "\n**الجينات المحددة:**\n"
            for gene in genes:
                if gene in self.knowledge['genes']:
                    gene_info = self.knowledge['genes'][gene]
                    explanation += f"• **{gene}:** {gene_info['inheritance']} - {gene_info['breeding_info']}\n"
        
        return explanation
    
    def _phenotype_description(self, genes):
        """وصف الأنماط الظاهرية"""
        description = "🎨 **الأنماط الظاهرية:**\n\n"
        
        if genes:
            for gene in genes:
                if gene in self.knowledge['genes']:
                    gene_info = self.knowledge['genes'][gene]
                    description += f"**{gene}:**\n{gene_info['phenotype']}\n\n"
                    
                    if gene_info['combinations']:
                        description += "**التركيبات المختلفة:**\n"
                        for combo, result in gene_info['combinations'].items():
                            description += f"• {combo} = {result}\n"
                        description += "\n"
        else:
            description += """
🌈 **ألوان الحمام الأساسية:**

• **الأزرق:** اللون الأساسي مع أنماط مختلفة
• **الأحمر:** من الوردي الفاتح إلى الأحمر الداكن
• **الأسود:** أزرق + جين Spread
• **البني:** لون شوكولاتي نادر
• **الأبيض:** يخفي جميع الألوان الأخرى

🎭 **الأنماط:**

• **الخطوط (Bar):** النمط الطبيعي الأكثر شيوعاً
• **الشطرنج (Checker):** مربعات متناوبة
• **الصلب (Spread):** لون موحد بدون أنماط
"""
        
        return description
    
    def _compare_genes(self, genes):
        """مقارنة الجينات"""
        if len(genes) < 2:
            return "لإجراء مقارنة، يرجى ذكر جينين أو أكثر."
        
        comparison = "⚖️ **مقارنة الجينات:**\n\n"
        
        for i, gene in enumerate(genes):
            if gene in self.knowledge['genes']:
                gene_info = self.knowledge['genes'][gene]
                comparison += f"**{i+1}. {gene}:**\n"
                comparison += f"• الكروموسوم: {gene_info['chromosome']}\n"
                comparison += f"• الوراثة: {gene_info['inheritance']}\n"
                comparison += f"• التأثير: {gene_info['phenotype']}\n\n"
        
        return comparison
    
    def _general_response(self, query):
        """الإجابة العامة للاستعلامات غير المصنفة"""
        query_lower = query.lower()
        
        # البحث في الأسئلة الشائعة
        for question, answer in self.knowledge['common_questions'].items():
            if any(word in query_lower for word in question.split()):
                return f"💡 **{question}**\n\n{answer}"
        
        # إجابة افتراضية
        return """
🤔 لم أفهم سؤالك تماماً. يمكنني مساعدتك في:

🧬 **معلومات الجينات:** اسأل عن أي جين مثل "ما هو جين Spread؟"
🐣 **نصائح التربية:** اسأل عن التزاوج والنسل
🎨 **الألوان والأنماط:** اسأل عن الأشكال والألوان
📚 **الوراثة:** اسأل عن كيفية انتقال الجينات

**أمثلة للأسئلة:**
• ما الفرق بين Blue و Ash-red؟
• كيف أحصل على حمام أسود صلب؟
• ما هو أفضل تزاوج للألوان الجميلة؟
• كيف تورث الجينات المرتبطة بالجنس؟
"""

# -------------------------------------------------
#  3. قاعدة البيانات والبحث
# -------------------------------------------------

@st.cache_resource
def init_sqlite_db():
    """إنشاء قاعدة بيانات SQLite"""
    db_path = os.path.join(tempfile.gettempdir(), "genetics_knowledge.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_query TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    return conn

@st.cache_resource
def get_ai_agent():
    """الحصول على وكيل الذكاء الاصطناعي"""
    return GeneticsAI()

# -------------------------------------------------
#  4. واجهة المستخدم التفاعلية
# -------------------------------------------------

def main():
    st.title("🕊️ العرّاب للجينات - الوكيل الذكي التفاعلي")
    st.markdown("*نظام ذكي متقدم لاستكشاف وراثة الحمام مع محادثة تفاعلية*")

    # تحميل المكونات
    try:
        db_conn = init_sqlite_db()
        ai_agent = get_ai_agent()
        
        # الشريط الجانبي
        with st.sidebar:
            st.header("🧠 حالة الوكيل الذكي")
            st.success("متصل ومستعد للمحادثة")
            
            st.header("📊 إحصائيات")
            cursor = db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_history")
            chat_count = cursor.fetchone()[0]
            st.metric("عدد المحادثات", chat_count)
            
            st.header("🧬 الجينات المتاحة")
            for gene_name in GENETICS_KNOWLEDGE["genes"].keys():
                st.write(f"• {gene_name}")
            
            # مسح التاريخ
            if st.button("🗑️ مسح تاريخ المحادثة"):
                st.session_state.messages = []
                st.rerun()

        # التبويبات
        tab1, tab2, tab3 = st.tabs(["🤖 المحادثة التفاعلية", "🧬 موسوعة الجينات", "📊 إحصائيات المحادثة"])

        with tab1:
            st.header("محادثة مع الخبير الوراثي")
            
            # تهيئة تاريخ المحادثة
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "مرحباً! أنا الوكيل الذكي لوراثة الحمام. يمكنني الإجابة على أسئلتك حول الجينات والتربية والألوان. ما الذي تود معرفته؟"}
                ]

            # عرض تاريخ المحادثة
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # أمثلة سريعة
            st.markdown("**أمثلة سريعة:**")
            example_cols = st.columns(4)
            
            examples = [
                "ما هو جين Spread؟",
                "كيف أحصل على حمام أسود؟", 
                "ما الفرق بين Blue و Ash-red؟",
                "نصائح للتربية"
            ]
            
            for i, example in enumerate(examples):
                with example_cols[i]:
                    if st.button(example, key=f"example_{i}"):
                        # إضافة السؤال النموذجي
                        st.session_state.messages.append({"role": "user", "content": example})
                        
                        # الحصول على الإجابة
                        with st.spinner("جاري التفكير..."):
                            response = ai_agent.generate_response(example)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # حفظ في قاعدة البيانات
                        cursor = db_conn.cursor()
                        cursor.execute("""
                            INSERT INTO chat_history (user_query, ai_response)
                            VALUES (?, ?)
                        """, (example, response))
                        db_conn.commit()
                        
                        st.rerun()

            # حقل الإدخال للمحادثة
            if prompt := st.chat_input("اكتب سؤالك هنا..."):
                # إضافة رسالة المستخدم
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)

                # الحصول على رد الوكيل الذكي
                with st.chat_message("assistant"):
                    with st.spinner("جاري التفكير..."):
                        response = ai_agent.generate_response(prompt)
                    st.markdown(response)

                # إضافة رد الوكيل لتاريخ المحادثة
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # حفظ في قاعدة البيانات
                cursor = db_conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_history (user_query, ai_response)
                    VALUES (?, ?)
                """, (prompt, response))
                db_conn.commit()

        with tab2:
            st.header("موسوعة الجينات الشاملة")
            
            # فلترة الجينات
            selected_inheritance = st.selectbox(
                "فلترة حسب نوع الوراثة:",
                ["الكل", "Sex-linked", "Autosomal"]
            )
            
            # عرض الجينات
            for gene_name, gene_info in GENETICS_KNOWLEDGE["genes"].items():
                if selected_inheritance == "الكل" or gene_info["inheritance"] == selected_inheritance:
                    with st.expander(f"🧬 {gene_name} ({gene_info['symbol']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**المعلومات الأساسية:**")
                            st.write(f"🔸 **الرمز:** `{gene_info['symbol']}`")
                            st.write(f"🔸 **الكروموسوم:** {gene_info['chromosome']}")
                            st.write(f"🔸 **نوع الوراثة:** {gene_info['inheritance']}")
                            
                            st.markdown("**النمط الظاهري:**")
                            st.write(gene_info['phenotype'])
                        
                        with col2:
                            st.markdown("**الوصف:**")
                            st.write(gene_info['description'])
                            
                            st.markdown("**معلومات التربية:**")
                            st.write(gene_info['breeding_info'])
                        
                        st.markdown("**التركيبات الشائعة:**")
                        for combo, result in gene_info['combinations'].items():
                            st.write(f"• **{combo}** → {result}")

        with tab3:
            st.header("إحصائيات وتحليلات المحادثة")
            
            cursor = db_conn.cursor()
            cursor.execute("""
                SELECT user_query, ai_response, timestamp 
                FROM chat_history 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            recent_chats = cursor.fetchall()
            
            if recent_chats:
                st.subheader("آخر المحادثات")
                for i, (query, response, timestamp) in enumerate(recent_chats):
                    with st.expander(f"محادثة {i+1}: {query[:50]}... ({timestamp})"):
                        st.write("**السؤال:**", query)
                        st.write("**الإجابة:**", response[:200] + "..." if len(response) > 200 else response)
            else:
                st.info("لا توجد محادثات سابقة")
            
            # إحصائيات إضافية
            cursor.execute("SELECT COUNT(*) FROM chat_history WHERE date(timestamp) = date('now')")
            today_count = cursor.fetchone()[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("محادثات اليوم", today_count)
            with col2:
                st.metric("إجمالي المحادثات", len(recent_chats))
            with col3:
                st.metric("الجينات المتاحة", len(GENETICS_KNOWLEDGE["genes"]))

    except Exception as e:
        st.error(f"خطأ في تشغيل النظام: {str(e)}")
        st.info("يتم تشغيل النظام في الوضع الآمن.")

if __name__ == "__main__":
    main()
