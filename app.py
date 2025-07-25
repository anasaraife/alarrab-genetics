# ===================================================================
# 🚀 العرّاب للجينات V56.0 - الوكيل الخبير النهائي
# يجمع بين ذاكرة الكتاب الكاملة وسرعة البث المباشر للإجابة
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
st.set_page_config(layout="wide", page_title="العرّاب للجينات - الوكيل الخبير")

# --- 2. تحميل المكتبات الاختيارية عند الحاجة ---
@st.cache_resource
def import_langchain():
    """
    Imports heavy langchain libraries only when needed.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    return GoogleGenerativeAIEmbeddings, FAISS, ChatGoogleGenerativeAI, RetrievalQA, PromptTemplate

# --- 3. قاعدة البيانات الوراثية (للحاسبة) ---
GENE_DATA = {
    'B': {
        'display_name_ar': "اللون الأساسي", 'type_en': 'sex-linked',
        'alleles': { 'BA': {'name': 'آش ريد'}, '+': {'name': 'أزرق/أسود'}, 'b': {'name': 'بني'} },
        'dominance': ['BA', '+', 'b']
    },
    'd': {
        'display_name_ar': "التخفيف", 'type_en': 'sex-linked',
        'alleles': { '+': {'name': 'عادي (غير مخفف)'}, 'd': {'name': 'مخفف'} },
        'dominance': ['+', 'd']
    },
    'e': {
        'display_name_ar': "أحمر متنحي", 'type_en': 'autosomal',
        'alleles': { '+': {'name': 'عادي (غير أحمر متنحي)'}, 'e': {'name': 'أحمر متنحي'} },
        'dominance': ['+', 'e']
    },
    'C': {
        'display_name_ar': "النمط", 'type_en': 'autosomal',
        'alleles': { 'CT': {'name': 'نمط تي (مخملي)'}, 'C': {'name': 'تشيكر'}, '+': {'name': 'بار (شريط)'}, 'c': {'name': 'بدون شريط'} },
        'dominance': ['CT', 'C', '+', 'c']
    },
    'S': {
        'display_name_ar': "الانتشار (سبريد)", 'type_en': 'autosomal',
        'alleles': { 'S': {'name': 'منتشر (سبريد)'}, '+': {'name': 'عادي (غير منتشر)'} },
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
            phenotypes['B'] = 'أحمر متنحي'; phenotypes['C'] = ''
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

# --- 5. وظائف المساعد الذكي الخبير (Agent) ---
@st.cache_resource
def load_knowledge_base():
    """
    تحميل ذاكرة الوكيل (قاعدة البيانات المتجهة) من الملفات.
    """
    try:
        GoogleGenerativeAIEmbeddings, FAISS, _, _, _ = import_langchain()
        if "GEMINI_API_KEY" not in st.secrets:
            return None, "مفتاح Google API غير موجود في الأسرار (Secrets)."
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db_path = "faiss_index_pigeon_genetics"
        if not os.path.exists(db_path):
            return None, f"لم يتم العثور على مجلد قاعدة البيانات '{db_path}'. يرجى اتباع دليل التحديث."
        vector_db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        return vector_db, None
    except Exception as e:
        return None, f"حدث خطأ أثناء تحميل قاعدة المعرفة: {e}"

def ask_expert_agent_stream(query, db):
    """
    تبث الإجابة مباشرة كلمة بكلمة بناءً على البحث في ذاكرة الكتاب.
    """
    try:
        _, _, ChatGoogleGenerativeAI, _, PromptTemplate = import_langchain()
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        # 1. البحث عن المعلومات ذات الصلة في الكتاب
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(query)
        
        context = ""
        for i, doc in enumerate(relevant_docs):
            context += f"مصدر {i+1}:\n{doc.page_content}\n\n"

        # 2. بناء البرومبت النهائي
        template = """
        أنت "العرّاب الذكي"، خبير عالمي في وراثة الحمام. مهمتك هي الإجابة على سؤال المستخدم بدقة وعمق بناءً على المعلومات المتوفرة في المصادر فقط.
        
        **المصادر من الكتاب:**
        ---
        {context}
        ---

        **سؤال المستخدم:** {question}

        **تعليمات:**
        1. أجب باللغة العربية بأسلوب علمي ومفصل.
        2. استخدم المعلومات من المصادر المتوفرة فقط. لا تخترع أي معلومات.
        3. إذا كانت المصادر لا تحتوي على إجابة واضحة، قل "المعلومات المتوفرة في المصدر الحالي لا تجيب على هذا السؤال بشكل مباشر، ولكن يمكن استنتاج ما يلي...".
        4. إذا كان السؤال ترحيباً أو عاماً جداً، أجب بشكل ودي ومختصر.

        **الإجابة المفصلة:**
        """
        
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        chain = prompt | llm
        
        # 3. بث الإجابة مباشرة
        for chunk in chain.stream({"context": context, "question": query}):
            yield chunk.content

    except Exception as e:
        yield f"❌ حدث خطأ تقني: {str(e)[:200]}..."

# --- 6. واجهة التطبيق ---
st.title("🕊️ العرّاب للجينات (V56 - الوكيل الخبير النهائي)")

vector_db, error_message = load_knowledge_base()

tab1, tab2 = st.tabs(["🤖 المساعد الخبير", "🧬 الحاسبة الوراثية"])

with tab1:
    st.header("💬 تحدث مع الخبير")
    
    if error_message:
        st.error(f"**خطأ في تحميل قاعدة المعرفة:** {error_message}")
        st.warning("لن يتمكن المساعد الخبير من العمل حتى يتم حل هذه المشكلة.")
    elif vector_db is None:
        st.warning("جاري تحميل قاعدة المعرفة، يرجى الانتظار...")
    else:
        st.success("✅ قاعدة المعرفة جاهزة. يمكنك الآن طرح أي سؤال حول محتوى الكتاب.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("مثال: اشرح لي عن جين الأوبال..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = response_container.write_stream(ask_expert_agent_stream(prompt, vector_db))
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

with tab2:
    st.header("🧬 الحاسبة الوراثية السريعة")
    parent_inputs = {'male': {}, 'female': {}}
    input_col, result_col = st.columns([2, 3])
    with input_col:
        st.subheader("📝 إدخال البيانات")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**♂️ الذكر (الأب)**")
            for gene, data in GENE_DATA.items():
                with st.expander(f"{data['display_name_ar']}"):
                    choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
                    parent_inputs['male'][f'{gene}_visible'] = st.selectbox("الصفة الظاهرية", choices, key=f"male_{gene}_visible")
                    parent_inputs['male'][f'{gene}_hidden'] = st.selectbox("الصفة الخفية", choices, key=f"male_{gene}_hidden")
        with col2:
            st.write("**♀️ الأنثى (الأم)**")
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
        st.subheader("📊 النتائج المتوقعة")
        if st.button("⚡ احسب الآن", use_container_width=True, type="primary"):
            if not all([parent_inputs['male'].get('B_visible') != "(لا اختيار)", parent_inputs['female'].get('B_visible') != "(لا اختيار)"]):
                st.error("⚠️ الرجاء اختيار اللون الأساسي لكلا الوالدين.")
            else:
                with st.spinner("🧮 جاري الحساب..."):
                    results = predict_genetics_final(parent_inputs)
                    total = sum(results.values())
                    st.success(f"✅ تم حساب {total} تركيبة!")
                    df_results = pd.DataFrame([{'النمط الظاهري': p, 'النمط الوراثي': g, 'النسبة %': f"{(c/total)*100:.1f}%"} for (p, g), c in results.items()])
                    st.dataframe(df_results, use_container_width=True)
                    chart_data = df_results.set_index('النمط الظاهري')['النسبة %'].str.rstrip('%').astype('float')
                    st.bar_chart(chart_data)
