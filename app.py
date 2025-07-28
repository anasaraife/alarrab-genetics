# ==============================================================================
#  العرّاب للجينات - الإصدار 7.0 (الوكيل الذكي المستقل)
#  - يعتمد على قاعدة معرفة داخلية لضمان السرعة والاستقرار.
# ==============================================================================

import streamlit as st
import sqlite3
import os
import tempfile
from datetime import datetime

# -------------------------------------------------
#  1. إعدادات الصفحة وقاعدة المعرفة الداخلية
# -------------------------------------------------
st.set_page_config(
    page_title="العرّاب للجينات - الوكيل الذكي",
    page_icon="🕊️",
    layout="wide",
)

# قاعدة المعرفة الداخلية الموسعة (عقل الوكيل)
GENETICS_KNOWLEDGE = {
    "genes": {
        "Blue/Black": {
            "symbol": "B+", "chromosome": "Z", "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون الأزرق/الأسود في الحمام. هذا الجين سائد ومرتبط بالجنس.",
            "phenotype": "لون أزرق رمادي أو أسود حسب وجود جينات أخرى مثل Spread",
            "breeding_info": "الذكور يحتاجون نسختين من الجين، الإناث نسخة واحدة فقط",
            "combinations": {"B+ + S": "أسود صلب", "B+ without S": "أزرق مع أنماط", "B+ + C": "أزرق شطرنج"}
        },
        "Ash-red": {
            "symbol": "BA", "chromosome": "Z", "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون الأحمر الرمادي. سائد على Blue ومتنحي أمام Brown في بعض الحالات.",
            "phenotype": "لون أحمر رمادي مع تدرجات مختلفة من الوردي إلى الأحمر الداكن",
            "breeding_info": "ينتج ألوان جميلة عند التزاوج مع Blue",
            "combinations": {"BA + S": "أحمر صلب", "BA + C": "أحمر شطرنج", "BA + T": "أحمر مع خطوط"}
        },
        "Brown": {
            "symbol": "b", "chromosome": "Z", "inheritance": "Sex-linked",
            "description": "الجين المسؤول عن اللون البني. متنحي أمام Blue و Ash-red.",
            "phenotype": "لون بني شوكولاتي عميق",
            "breeding_info": "نادر الظهور، يحتاج والدين حاملين للجين",
            "combinations": {"b + S": "بني صلب", "b + C": "بني شطرنج"}
        },
        "Checker": {
            "symbol": "C", "chromosome": "1", "inheritance": "Autosomal",
            "description": "نمط الشطرنج على الأجنحة. سائد جزئياً على T-pattern.",
            "phenotype": "نمط مربعات داكنة وفاتحة على الأجنحة يشبه رقعة الشطرنج",
            "breeding_info": "يظهر في كلا الجنسين بنفس الطريقة",
            "combinations": {"C + Blue": "شطرنج أزرق", "C + Ash-red": "شطرنج أحمر"}
        },
        "Spread": {
            "symbol": "S", "chromosome": "8", "inheritance": "Autosomal",
            "description": "انتشار اللون على كامل الطائر. يخفي جميع الأنماط الأخرى.",
            "phenotype": "لون موحد بدون أنماط أو خطوط على كامل الجسم",
            "breeding_info": "سائد، يحتاج نسخة واحدة فقط للظهور",
            "combinations": {"S + أي لون": "لون صلب بدون أنماط"}
        },
        "Red Bar": {
            "symbol": "T", "chromosome": "1", "inheritance": "Autosomal",
            "description": "الخطوط الحمراء على الأجنحة. البديل البري الأساسي.",
            "phenotype": "خطان أحمران عرضيان على كل جناح",
            "breeding_info": "النمط الأساسي الأكثر شيوعاً",
            "combinations": {"T + Blue": "أزرق مع خطوط", "T + Ash-red": "أحمر مع خطوط داكنة"}
        }
    },
    "breeding_patterns": {
        "sex_linked": "الجينات المرتبطة بالجنس تورث من الأب للبنات ومن الأم للأولاد.",
        "autosomal": "الجينات الجسمية تورث بنفس الطريقة في كلا الجنسين.",
        "dominance": "الجين السائد يظهر حتى لو كان موجود في نسخة واحدة فقط."
    },
    "common_questions": {
        "كيف أعرف جينات حمامتي": "يمكن معرفة الجينات من خلال اللون والنمط الظاهر، ولكن الاختبار الوراثي أدق.",
        "ما أفضل تزاوج للحصول على ألوان جميلة": "تزاوج Ash-red مع Blue ينتج تنوع جميل في الألوان.",
        "لماذا لا تظهر بعض الألوان في النسل": "قد تكون الجينات متنحية أو مخفية بواسطة جينات أخرى مثل Spread."
    }
}

# -------------------------------------------------
#  2. نظام الذكاء الاصطناعي للمحادثة
# -------------------------------------------------

class GeneticsAI:
    def __init__(self):
        self.knowledge = GENETICS_KNOWLEDGE
    
    def analyze_query(self, query):
        query_lower = query.lower()
        question_types = {
            'gene_info': ['ما هو', 'اشرح', 'معلومات عن', 'تعريف'],
            'breeding': ['تزاوج', 'تربية', 'نسل', 'breeding', 'offspring'],
            'inheritance': ['وراثة', 'كيف يورث', 'inheritance', 'inherit'],
            'phenotype': ['لون', 'شكل', 'مظهر', 'نمط', 'color', 'pattern'],
            'comparison': ['مقارنة', 'فرق', 'أفضل', 'compare', 'difference']
        }
        
        detected_types = [q_type for q_type, keywords in question_types.items() if any(keyword in query_lower for keyword in keywords)]
        
        mentioned_genes = [gene_name for gene_name in self.knowledge['genes'].keys() if gene_name.lower() in query_lower or any(keyword in query_lower for keyword in (gene_name.split('/')[0].lower(), gene_name.split('/')[-1].lower() if '/' in gene_name else gene_name.lower()))]
        
        return {'types': detected_types, 'genes': mentioned_genes}

    def generate_response(self, query):
        analysis = self.analyze_query(query)
        
        if 'gene_info' in analysis['types'] and analysis['genes']:
            return self._explain_genes(analysis['genes'])
        elif 'breeding' in analysis['types']:
            return self._breeding_advice(analysis['genes'])
        elif 'inheritance' in analysis['types']:
            return self._inheritance_explanation(analysis['genes'])
        elif 'phenotype' in analysis['types']:
            return self._phenotype_description(analysis['genes'])
        elif 'comparison' in analysis['types']:
            return self._compare_genes(analysis['genes'])
        else:
            return self._general_response(query)

    def _explain_genes(self, genes):
        if not genes: return "لم أستطع تحديد الجين المقصود. يرجى ذكر اسم الجين بوضوح."
        explanations = []
        for gene in genes:
            if gene in self.knowledge['genes']:
                gene_info = self.knowledge['genes'][gene]
                explanation = f"🧬 **{gene} ({gene_info['symbol']})**\n\n"
                explanation += f"📍 **الموقع:** الكروموسوم {gene_info['chromosome']}\n"
                explanation += f"🔄 **نوع الوراثة:** {gene_info['inheritance']}\n\n"
                explanation += f"📝 **الوصف:**\n{gene_info['description']}\n\n"
                explanation += f"🎨 **النمط الظاهري:**\n{gene_info['phenotype']}\n\n"
                explanation += f"🐣 **معلومات التربية:**\n{gene_info['breeding_info']}\n\n"
                explanation += "🔀 **التركيبات الشائعة:**\n"
                for combo, result in gene_info['combinations'].items():
                    explanation += f"• {combo} → {result}\n"
                explanations.append(explanation)
        return "\n\n".join(explanations)

    def _breeding_advice(self, genes):
        advice = "💡 **نصائح التربية والتزاوج:**\n\n"
        if genes:
            for gene in genes:
                if gene in self.knowledge['genes']:
                    advice += f"**{gene}:** {self.knowledge['genes'][gene]['breeding_info']}\n\n"
        else:
            advice += "1. **للحصول على ألوان متنوعة:** جرب تزاوج Ash-red مع Blue.\n"
            advice += "2. **للألوان الصلبة:** استخدم الحمام الحامل لجين Spread.\n"
            advice += "3. **للأنماط الجميلة:** تجنب استخدام Spread إذا كنت تريد رؤية الأنماط.\n"
            advice += "4. **للجينات المرتبطة بالجنس:** الذكر يحدد لون الإناث، والأنثى تحدد لون الذكور.\n"
        return advice

    def _inheritance_explanation(self, genes):
        explanation = "🔬 **أنماط الوراثة في الحمام:**\n\n"
        explanation += f"**الوراثة المرتبطة بالجنس (Sex-linked):**\n{self.knowledge['breeding_patterns']['sex_linked']}\n*أمثلة: Blue/Black, Ash-red, Brown*\n\n"
        explanation += f"**الوراثة الجسمية (Autosomal):**\n{self.knowledge['breeding_patterns']['autosomal']}\n*أمثلة: Checker, Spread, Red Bar*\n\n"
        if genes:
            explanation += "\n**الجينات المحددة:**\n"
            for gene in genes:
                if gene in self.knowledge['genes']:
                    gene_info = self.knowledge['genes'][gene]
                    explanation += f"• **{gene}:** {gene_info['inheritance']} - {gene_info['breeding_info']}\n"
        return explanation

    def _phenotype_description(self, genes):
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
            description += "🌈 **ألوان الحمام الأساسية:**\n• الأزرق، الأحمر، الأسود، البني، الأبيض.\n\n"
            description += "🎭 **الأنماط:**\n• الخطوط (Bar)، الشطرنج (Checker)، الصلب (Spread)."
        return description

    def _compare_genes(self, genes):
        if len(genes) < 2: return "لإجراء مقارنة، يرجى ذكر جينين أو أكثر."
        comparison = "⚖️ **مقارنة الجينات:**\n\n"
        for i, gene in enumerate(genes):
            if gene in self.knowledge['genes']:
                gene_info = self.knowledge['genes'][gene]
                comparison += f"**{i+1}. {gene}:**\n"
                comparison += f"• الوراثة: {gene_info['inheritance']}\n"
                comparison += f"• التأثير: {gene_info['phenotype']}\n\n"
        return comparison

    def _general_response(self, query):
        for question, answer in self.knowledge['common_questions'].items():
            if any(word in query.lower() for word in question.split()):
                return f"💡 **{question}**\n\n{answer}"
        return "🤔 لم أفهم سؤالك تماماً. يمكنك أن تسأل عن معلومات جين معين، نصائح للتربية، أو مقارنة بين جينين."

# -------------------------------------------------
#  3. قاعدة البيانات والواجهة
# -------------------------------------------------

@st.cache_resource
def init_sqlite_db():
    """إنشاء قاعدة بيانات SQLite لحفظ سجل المحادثات."""
    db_path = os.path.join(tempfile.gettempdir(), "chat_history.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY, user_query TEXT, ai_response TEXT, timestamp TIMESTAMP
        )
    """)
    conn.commit()
    return conn

@st.cache_resource
def get_ai_agent():
    """الحصول على نسخة واحدة من الوكيل الذكي."""
    return GeneticsAI()

# --- الواجهة الرئيسية ---
st.title("🕊️ العرّاب للجينات - الوكيل الذكي المستقل")
st.markdown("*نظام حواري متقدم لاستكشاف وراثة الحمام*")

db_conn = init_sqlite_db()
ai_agent = get_ai_agent()

with st.sidebar:
    st.header("🧠 حالة الوكيل")
    st.success("متصل وجاهز")
    st.header("🧬 الجينات الأساسية")
    for gene_name in GENETICS_KNOWLEDGE["genes"].keys():
        st.write(f"• {gene_name}")
    if st.button("🗑️ مسح المحادثة الحالية"):
        st.session_state.messages = []
        st.rerun()

# تهيئة وعرض المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "مرحباً! أنا العرّاب. كيف يمكنني مساعدتك في عالم وراثة الحمام اليوم؟"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# إدخال المستخدم
if prompt := st.chat_input("اسأل عن جين، تزاوج، أو لون..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("العرّاب يفكر..."):
            response = ai_agent.generate_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # حفظ في قاعدة البيانات
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_query, ai_response, timestamp) VALUES (?, ?, ?)",
                   (prompt, response, datetime.now()))
    db_conn.commit()
