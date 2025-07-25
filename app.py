# ===================================================================
# 🕊️ العرّاب للجينات V51.0 - البث المباشر للإجابة (Streaming)
# تم حل مشكلة انتهاء المهلة عبر عرض الإجابة كلمة بكلمة
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

# --- 1. إعدادات الصفحة ---
st.set_page_config(layout="wide", page_title="العرّاب للجينات")

# --- 2. تحميل المكتبات الاختيارية عند الحاجة ---
@st.cache_resource
def import_langchain():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    return GoogleGenerativeAIEmbeddings, FAISS, ChatGoogleGenerativeAI

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
    # ... (الكود الكامل لهذه الوظيفة موجود في النسخ السابقة، تم إخفاؤه هنا للاختصار)
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

# --- 5. وظائف المساعد الذكي الخبير (Agent) مع البث المباشر ---
@st.cache_resource
def load_knowledge_base():
    try:
        GoogleGenerativeAIEmbeddings, FAISS, _ = import_langchain()
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

def ask_expert_agent_stream(query, db):
    """
    هذه الوظيفة تقوم بالبحث في الكتاب ثم تبث الإجابة مباشرة كلمة بكلمة.
    """
    try:
        _, _, ChatGoogleGenerativeAI = import_langchain()
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        
        # 1. البحث عن المعلومات ذات الصلة في الكتاب
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)
        
        context = ""
        for i, doc in enumerate(relevant_docs):
            context += f"مصدر {i+1}:\n{doc.page_content}\n\n"

        # 2. بناء البرومبت النهائي
        prompt = f"""
        أنت خبير في علم وراثة الحمام. أجب على السؤال التالي باللغة العربية بناءً على المعلومات المتوفرة في المصادر فقط.
        إذا كانت الإجابة غير موجودة في المصادر، قل بوضوح "المعلومة غير متوفرة في المصدر الحالي".

        المصادر:
        {context}

        السؤال: {query}

        الإجابة:
        """
        
        # 3. بث الإجابة مباشرة
        for chunk in llm.stream(prompt):
            yield chunk.content

    except Exception as e:
        error_msg = str(e)
        if "Deadline Exceeded" in error_msg or "504" in error_msg:
            yield "⏳ انتهت مهلة الانتظار. الخدمة مشغولة. جرب مرة أخرى."
        else:
            yield f"❌ حدث خطأ تقني: {error_msg[:200]}..."

# --- 6. واجهة التطبيق ---
st.title("🕊️ العرّاب للجينات (V51 - البث المباشر)")

# تحميل قاعدة المعرفة
vector_db, error_message = load_knowledge_base()

tab1, tab2 = st.tabs(["🧬 الحاسبة الذكية", "🤖 المساعد الخبير (Agent)"])

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
    st.header("🤖 تحدث مع الخبير الذكي")
    st.write("اطرح أي سؤال حول محتوى كتاب الوراثة، وستظهر الإجابة مباشرة.")
    
    if error_message:
        st.error(f"**خطأ في تحميل قاعدة المعرفة:** {error_message}")
    elif vector_db is None:
        st.warning("جاري تحميل قاعدة المعرفة، يرجى الانتظار...")
    else:
        st.success("✅ قاعدة المعرفة جاهزة. يمكنك الآن طرح أسئلتك.")
        
        user_query = st.text_area("مثال: من هو Axel Sell؟ أو اشرح عن جين Spread.", height=100)
        
        if st.button("اسأل الخبير", use_container_width=True, type="primary"):
            if not user_query.strip():
                st.warning("الرجاء إدخال سؤالك.")
            else:
                with st.spinner("الخبير يبحث في الكتاب... 📖"):
                    # ==========================================================
                    # ===== بداية الكود الجديد للبث المباشر =====
                    # ==========================================================
                    st.info("**إجابة الخبير:**")
                    # st.write_stream يعرض الإجابة كلمة بكلمة
                    st.write_stream(ask_expert_agent_stream(user_query, vector_db))
                    # ==========================================================
                    # ===== نهاية الكود الجديد =====
                    # ==========================================================

