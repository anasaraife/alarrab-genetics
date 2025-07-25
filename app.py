# ===================================================================
# 🕊️ العرّاب للجينات V50.0 - تحسينات الذكاء والسرعة والدقة
# تحسينات جديدة: ذاكرة محادثة، إجابات أذكى، استرجاع محسن، معالج أخطاء متقدم
# ===================================================================

import streamlit as st
from itertools import product
import collections
import pandas as pd
import google.generativeai as genai
import json
import os
import time
from datetime import datetime
import hashlib

# --- 1. إعدادات الصفحة ---
st.set_page_config(layout="wide", page_title="العرّاب للجينات")

# --- 2. تحميل المكتبات الاختيارية عند الحاجة ---
@st.cache_resource
def import_langchain():
    """
    Imports heavy langchain libraries only when needed.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI
    return GoogleGenerativeAIEmbeddings, FAISS, RetrievalQA, ChatGoogleGenerativeAI

# --- 3. قاعدة البيانات الوراثية (كما في السابق) ---
GENE_DATA = {
    'B': {
        'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked',
        'alleles': {
            'BA': {'name': 'آش ريد', 'is_recessive': False},
            '+': {'name': 'أزرق/أسود', 'is_recessive': False},
            'b': {'name': 'بني', 'is_recessive': True}
        },
        'dominance': ['BA', '+', 'b']
    },
    'd': {
        'display_name_ar': "التخفيف", 'type_en': 'sex-linked',
        'alleles': {
            '+': {'name': 'عادي (غير مخفف)', 'is_recessive': False},
            'd': {'name': 'مخفف', 'is_recessive': True}
        },
        'dominance': ['+', 'd']
    },
    'e': {
        'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal',
        'alleles': {
            '+': {'name': 'عادي (غير أحمر متنحي)', 'is_recessive': False},
            'e': {'name': 'أحمر متنحي', 'is_recessive': True}
        },
        'dominance': ['+', 'e']
    },
    'C': {
        'display_name_ar': "النمط", 'type_en': 'autosomal',
        'alleles': {
            'CT': {'name': 'نمط تي (مخملي)', 'is_recessive': False},
            'C': {'name': 'تشيكر', 'is_recessive': False},
            '+': {'name': 'بار (شريط)', 'is_recessive': False},
            'c': {'name': 'بدون شريط', 'is_recessive': True}
        },
        'dominance': ['CT', 'C', '+', 'c']
    },
    'S': {
        'display_name_ar': "الانتشار (سبريد)", 'type_en': 'autosomal',
        'alleles': {
            'S': {'name': 'منتشر (سبريد)', 'is_recessive': False},
            '+': {'name': 'عادي (غير منتشر)', 'is_recessive': False}
        },
        'dominance': ['S', '+']
    }
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}

# --- 4. المحرك الوراثي (بدون تغيير) ---
class GeneticCalculator:
    def describe_phenotype(self, genotype_dict):
        phenotypes = {gene: "" for gene in GENE_ORDER}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').split('//')
            for dominant_allele in GENE_DATA[gene_name]['dominance']:
                if dominant_allele in alleles:
                    phenotypes[gene_name] = GENE_DATA[gene_name]['alleles'][dominant_allele]['name']
                    break
        if 'e//e' in genotype_dict.get('e', ''):
            phenotypes['B'] = 'أحمر متنحي'
            phenotypes['C'] = ''
        if 'S' in genotype_dict.get('S', ''):
            if 'e//e' not in genotype_dict.get('e', ''):
                phenotypes['C'] = 'منتشر (سبريد)'
        sex = "أنثى" if any('•' in genotype_dict.get(g, '') for g, d in GENE_DATA.items() if d['type_en'] == 'sex-linked') else "ذكر"
        desc_parts = [phenotypes.get('B')]
        if phenotypes.get('C'): desc_parts.append(phenotypes.get('C'))
        gt_str_parts = [genotype_dict[gene].strip() for gene in GENE_ORDER]
        gt_str = " | ".join(gt_str_parts)
        final_phenotype = " ".join(filter(None, desc_parts))
        return f"{sex} {final_phenotype}", gt_str

def predict_genetics_final(parent_inputs):
    calculator = GeneticCalculator()
    parent_genotypes = {}
    for parent in ['male', 'female']:
        gt_parts = []
        for gene in GENE_ORDER:
            gene_info = GENE_DATA[gene]
            visible_name = parent_inputs[parent].get(f'{gene}_visible')
            hidden_name = parent_inputs[parent].get(f'{gene}_hidden', visible_name)
            wild_type_symbol = next((s for s, n in gene_info['alleles'].items() if '+' in s or '⁺' in s), gene_info['dominance'][0])
            visible_symbol = NAME_TO_SYMBOL_MAP[gene].get(visible_name, wild_type_symbol)
            hidden_symbol = NAME_TO_SYMBOL_MAP[gene].get(hidden_name, visible_symbol)
            if gene_info['type_en'] == 'sex-linked' and parent == 'female':
                gt_parts.append(f"•//{visible_symbol}")
            else:
                alleles = sorted([visible_symbol, hidden_symbol], key=lambda x: gene_info['dominance'].index(x))
                gt_parts.append(f"{alleles[0]}//{alleles[1]}")
        parent_genotypes[parent] = gt_parts
    def generate_gametes(genotype_parts, is_female):
        parts_for_product = []
        for i, gt_part in enumerate(genotype_parts):
            gene_name = GENE_ORDER[i]
            if GENE_DATA[gene_name]['type_en'] == 'sex-linked' and is_female:
                parts_for_product.append([gt_part.replace('•//','').strip()])
            else:
                parts_for_product.append(gt_part.split('//'))
        return list(product(*parts_for_product))
    male_gametes = generate_gametes(parent_genotypes['male'], is_female=False)
    female_gametes = generate_gametes(parent_genotypes['female'], is_female=True)
    offspring_counts = collections.Counter()
    for m_gamete in male_gametes:
        for f_gamete in female_gametes:
            son_dict, daughter_dict = {}, {}
            for i, gene in enumerate(GENE_ORDER):
                alleles = sorted([m_gamete[i], f_gamete[i]], key=lambda x: GENE_DATA[gene]['dominance'].index(x))
                if GENE_DATA[gene]['type_en'] == 'sex-linked':
                    son_dict[gene], daughter_dict[gene] = f"{alleles[0]}//{alleles[1]}", f"•//{m_gamete[i]}"
                else:
                    gt_part = f"{alleles[0]}//{alleles[1]}"
                    son_dict[gene], daughter_dict[gene] = gt_part, gt_part
            offspring_counts[calculator.describe_phenotype(son_dict)] += 1
            offspring_counts[calculator.describe_phenotype(daughter_dict)] += 1
    return offspring_counts

# --- 5. نظام ذاكرة المحادثة المحسن ---
def add_to_memory(question, answer):
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = []
    
    st.session_state.conversation_memory.append({
        'timestamp': datetime.now().strftime("%H:%M"),
        'question': question,
        'answer': answer,
    })
    
    # حفظ آخر 10 محادثات فقط
    if len(st.session_state.conversation_memory) > 10:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-10:]

def get_conversation_context():
    if 'conversation_memory' not in st.session_state or not st.session_state.conversation_memory:
        return ""
    
    context = "هذه هي المحادثات الأخيرة في هذه الجلسة:\n"
    for item in st.session_state.conversation_memory[-3:]: # آخر 3 محادثات
        context += f"سؤال سابق: {item['question'][:100]}...\nجواب سابق: {item['answer'][:200]}...\n---\n"
    return context

# --- 6. وظائف المساعد الذكي المحسنة ---
@st.cache_resource
def load_knowledge_base():
    try:
        GoogleGenerativeAIEmbeddings, FAISS, _, _ = import_langchain()
        if "GEMINI_API_KEY" not in st.secrets:
            return None, "مفتاح Google API غير موجود في الأسرار (Secrets)."
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db_path = "faiss_index_pigeon_genetics"
        if not os.path.exists(db_path):
            return None, f"لم يتم العثور على مجلد قاعدة البيانات '{db_path}'."
        vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        return vector_db, None
    except Exception as e:
        return None, f"حدث خطأ أثناء تحميل قاعدة المعرفة: {e}"

def ask_expert_agent_enhanced(query, db):
    try:
        _, _, RetrievalQA, ChatGoogleGenerativeAI = import_langchain()
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            request_timeout=90,
        )
        
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        conversation_context = get_conversation_context()
        
        enhanced_prompt = f"""
        أنت خبير في علم وراثة الحمام. استخدم سياق المحادثة السابقة والمصادر التالية للإجابة على السؤال بدقة.

        {conversation_context}

        السؤال الحالي: {query}

        تعليمات:
        1. أجب باللغة العربية.
        2. إذا كان السؤال مرتبطاً بمحادثة سابقة، اربط الإجابة بالسياق.
        3. استخدم المعلومات من المصادر التي يتم تزويدك بها فقط.
        4. إذا لم تجد المعلومة، قل ذلك بوضوح.

        الإجابة:
        """
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
        )
        
        response = qa_chain.invoke(enhanced_prompt)
        answer = response.get('result', "لم أتمكن من العثور على إجابة.")
        
        return answer
        
    except Exception as e:
        error_msg = str(e)
        if "Deadline Exceeded" in error_msg or "504" in error_msg:
            return "⏳ انتهت مهلة الانتظار. الخدمة مشغولة. جرب مرة أخرى."
        elif "api key" in error_msg.lower():
            return "🔑 مشكلة في مفتاح API. تأكد من صحته."
        else:
            return f"❌ حدث خطأ تقني: {error_msg[:200]}..."

# --- 7. واجهة التطبيق ---
st.title("🕊️ العرّاب للجينات (V50 - الذكاء المحسن)")

# تحميل قاعدة المعرفة
vector_db, error_message = load_knowledge_base()

tab1, tab2 = st.tabs(["🧬 الحاسبة الذكية", "🤖 المساعد الخبير المطور"])

with tab1:
    # ... (الكود الخاص بالحاسبة الذكية كما هو) ...
    parent_inputs = {'male': {}, 'female': {}}
    input_col, result_col = st.columns([2, 3])
    with input_col:
        st.header("📝 المدخلات")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("♂️ الذكر (الأب)")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"male_{gene}_visible")
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"male_{gene}_hidden")
        with col2:
            st.subheader("♀️ الأنثى (الأم)")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['female'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"female_{gene}_visible")
                    if data['type_en'] != 'sex-linked':
                        parent_inputs['female'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"female_{gene}_hidden")
                    else:
                        st.info("لا يوجد صفة خفية")
                        parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']
    with result_col:
        st.header("📊 النتائج")
        if st.button("احسب النتائج", use_container_width=True, type="primary"):
            if not all([parent_inputs['male'].get('B_visible') != "(لا اختيار)", parent_inputs['female'].get('B_visible') != "(لا اختيار)"]):
                st.error("⚠️ الرجاء اختيار اللون الأساسي لكلا الوالدين.")
            else:
                with st.spinner("جاري حساب الاحتمالات..."):
                    results = predict_genetics_final(parent_inputs)
                    total = sum(results.values())
                    st.success(f"تم حساب {total} تركيبة محتملة!")
                    df = pd.DataFrame([{'التركيب': f"{p} ({g})", 'الاحتمالية': (c/total)*100} for (p,g),c in results.items()])
                    st.bar_chart(df.set_index('التركيب'))

with tab2:
    st.header("🤖 المساعد الخبير المطور")
    
    if error_message:
        st.error(f"**خطأ في تحميل قاعدة المعرفة:** {error_message}")
    elif vector_db is None:
        st.warning("جاري تحميل قاعدة المعرفة، يرجى الانتظار...")
    else:
        st.success("✅ قاعدة المعرفة جاهزة والمساعد الذكي في أفضل حالاته!")
        
        user_query = st.text_area("اطرح سؤالك هنا:", height=100, placeholder="مثال: اشرح لي عن جين Spread وتأثيره...")
        
        if st.button("🔍 اسأل الخبير", use_container_width=True, type="primary"):
            if not user_query.strip():
                st.warning("⚠️ الرجاء إدخال سؤالك.")
            else:
                with st.spinner("🔬 الخبير يحلل سؤالك ويبحث في المراجع..."):
                    answer = ask_expert_agent_enhanced(user_query, vector_db)
                    add_to_memory(user_query, answer)
                    st.info("✅ **إجابة الخبير:**")
                    st.write(answer)

        if 'conversation_memory' in st.session_state and st.session_state.conversation_memory:
            with st.expander("📜 سجل المحادثات الأخيرة", expanded=False):
                for item in reversed(st.session_state.conversation_memory):
                    st.write(f"**[{item['timestamp']}] س:** {item['question']}")
                    st.write(f"**ج:** {item['answer']}")
                    st.divider()

# --- 8. تذييل التطبيق ---
st.divider()
st.markdown("<div style='text-align: center; color: #666;'><p>🕊️ <strong>العرّاب للجينات V50.0</strong> - نظام ذكي متطور</p></div>", unsafe_allow_html=True)
