# ===================================================================
# 🕊️ العرّاب للجينات V38.0 - الحل النهائي
# تم إصلاح خطأ %%writefile باستخدام طريقة كتابة ملفات بايثون الأساسية
# ===================================================================

# 1. تثبيت المكتبات اللازمة
print("📦 جاري تثبيت Streamlit و pyngrok...")
!pip install streamlit pyngrok --quiet
print("✅ المكتبات جاهزة!")

# 2. تعريف كود التطبيق كسلسلة نصية واحدة
# This defines the entire application code as a single string variable
app_code = """
import streamlit as st
from itertools import product
import collections
import pandas as pd

# --- قاعدة البيانات الوراثية الكاملة ---
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
    },
    'Gr': {
        'display_name_ar': "الجريزل", 'type_en': 'autosomal',
        'alleles': {
            'Gr': {'name': 'جريزل', 'is_recessive': False},
            '+': {'name': 'عادي (غير جريزل)', 'is_recessive': False}
        },
        'dominance': ['Gr', '+']
    },
    'Op': {
        'display_name_ar': "الأوبال", 'type_en': 'autosomal',
        'alleles': {
            '+': {'name': 'عادي (غير أوبال)', 'is_recessive': False},
            'Op': {'name': 'أوبال', 'is_recessive': True}
        },
        'dominance': ['+', 'Op']
    },
    'My': {
        'display_name_ar': "الملكي", 'type_en': 'autosomal',
        'alleles': {
            '+': {'name': 'عادي (غير ملكي)', 'is_recessive': False},
            'My': {'name': 'ملكي', 'is_recessive': True}
        },
        'dominance': ['+', 'My']
    },
    'In': {
        'display_name_ar': "الإنديغو", 'type_en': 'autosomal',
        'alleles': {
            'In': {'name': 'إنديغو', 'is_recessive': False},
            '+': {'name': 'عادي (غير إنديغو)', 'is_recessive': False}
        },
        'dominance': ['In', '+']
    }
}
GENE_ORDER = list(GENE_DATA.keys())
NAME_TO_SYMBOL_MAP = {
    gene: {info['name']: symbol for symbol, info in data['alleles'].items()}
    for gene, data in GENE_DATA.items()
}

# --- المحرك الوراثي ---
class GeneticCalculator:
    def describe_phenotype(self, genotype_dict):
        phenotypes = {gene: "" for gene in GENE_ORDER}
        for gene_name, gt_part in genotype_dict.items():
            alleles = gt_part.replace('•//', '').replace('//', '')
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
        for gene in GENE_ORDER:
            phenotype_name = phenotypes.get(gene)
            if gene not in ['B', 'C', 'S', 'e'] and phenotype_name and "عادي" not in phenotype_name:
                desc_parts.append(phenotype_name)
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

# --- واجهة التطبيق باستخدام Streamlit ---
st.set_page_config(layout="wide", page_title="العرّاب للجينات")
st.title("🕊️ العرّاب للجينات (V38 - النسخة الكاملة)")

parent_inputs = {'male': {}, 'female': {}}

col1, col2 = st.columns(2)

with col1:
    st.header("🐦 معلومات الذكر (الأب)")
    for gene, data in GENE_DATA.items():
        with st.expander(f"{data['display_name_ar']}"):
            choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
            parent_inputs['male'][f'{gene}_visible'] = st.selectbox(f"الصفة الظاهرية", choices, key=f"male_{gene}_visible")
            parent_inputs['male'][f'{gene}_hidden'] = st.selectbox(f"الصفة الخفية (المحمول)", choices, key=f"male_{gene}_hidden")

with col2:
    st.header("🐦 معلومات الأنثى (الأم)")
    for gene, data in GENE_DATA.items():
        with st.expander(f"{data['display_name_ar']}"):
            choices = ["(لا اختيار)"] + [v['name'] for v in data['alleles'].values()]
            parent_inputs['female'][f'{gene}_visible'] = st.selectbox(f"الصفة الظاهرية", choices, key=f"female_{gene}_visible")
            if data['type_en'] != 'sex-linked':
                parent_inputs['female'][f'{gene}_hidden'] = st.selectbox(f"الصفة الخفية (المحمول)", choices, key=f"female_{gene}_hidden")
            else:
                st.info("لا يوجد صفة خفية (مرتبط بالجنس)")
                parent_inputs['female'][f'{gene}_hidden'] = parent_inputs['female'][f'{gene}_visible']


if st.button("📊 احسب النتائج", use_container_width=True):
    if not all([parent_inputs['male'].get('B_visible') != "(لا اختيار)", parent_inputs['female'].get('B_visible') != "(لا اختيار)"]):
        st.error("⚠️ الرجاء اختيار اللون الأساسي لكلا الوالدين.")
    else:
        results = predict_genetics_final(parent_inputs)
        total = sum(results.values())
        
        st.subheader(f"🧬 النتائج الوراثية ({total} تركيبة محتملة)")
        
        chart_data = []
        for (phenotype, genotype), count in sorted(results.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            st.write(f"- **{percentage:.2f}%** - {phenotype} `{genotype}`")
            chart_data.append({'التركيب المحتمل': f"{phenotype} ({genotype})", 'الاحتمالية': percentage})
        
        if chart_data:
            df = pd.DataFrame(chart_data)
            st.bar_chart(df.set_index('التركيب المحتمل'))
"""

# 3. كتابة الكود إلى ملف app.py باستخدام أوامر بايثون الأساسية
# This method is more reliable than %%writefile
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("✅ تم إنشاء ملف التطبيق app.py بنجاح.")


# 4. تشغيل التطبيق وإنشاء رابط عام
from pyngrok import ngrok
import os
import time

# إعداد رمز ngrok
NGROK_AUTHTOKEN = "30NOq6S2Ecs4tv1MdFQgsoYqgiG_2RmgZqwUKk2kwy9uXSvhR"
if NGROK_AUTHTOKEN and NGROK_AUTHTOKEN != "الصق رمز الدخول الخاص بك هنا":
    # نقتل أي عمليات ngrok قديمة لضمان بداية نظيفة
    ngrok.kill()
    os.system(f"ngrok config add-authtoken {NGROK_AUTHTOKEN}")
    print("✅ تم إعداد رمز ngrok.")

# تشغيل Streamlit في الخلفية
# نستخدم nohup لضمان استمرار عمله
!nohup streamlit run app.py --server.port 8501 &

# فتح نفق ngrok
# نعطي Streamlit بضع ثوان ليبدأ قبل فتح النفق
print("⏳ ننتظر قليلاً لبدء تشغيل Streamlit...")
time.sleep(5)

try:
    public_url = ngrok.connect(8501)
    print("====================================================================")
    print("✅✅✅ رابط التطبيق يعمل الآن ✅✅✅")
    print(public_url)
    print("====================================================================")
except Exception as e:
    print(f"❌ فشل تشغيل ngrok. الخطأ: {e}")

