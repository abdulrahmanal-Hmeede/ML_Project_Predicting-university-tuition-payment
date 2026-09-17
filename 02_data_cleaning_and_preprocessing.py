# %% [1] استيراد المكتبات البرمجية
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# %% [2] تحميل البيانات الأصلية وعزل نسخة العمل
# التأكد من وجود مجلد البيانات وقراءة ملف الإكسل
data_path = "مجموعة البيانات.xlsx"
if not os.path.exists(data_path):
    data_path = os.path.join("data", "مجموعة البيانات.xlsx")

df_raw = pd.read_excel(data_path)
df = df_raw.copy()

print(f"الأبعاد الأصلية للبيانات: {df.shape[0]} صف و {df.shape[1]} عمود.")

# %% [3] تنظيف أسماء الأعمدة وحذف الأعمدة غير المجدية
# إزالة المسافات الزائدة في بدايات ونهايات أسماء الأعمدة
df.columns = df.columns.str.strip()

# 1. حذف عمود الطابع الزمني لعدم وجود قيمة تنبؤية له في نماذج التعلم الآلي
# 2. حذف عمود قيمة القسط التقريبية الفارغ بنسبة 100% (153 قيمة مفقودة من 153)
cols_to_remove = ["طابع زمني", "قيمة القسط التقريببة"]
df = df.drop(columns=cols_to_remove, errors="ignore")

# %% [4] تنظيف النصوص وتوحيد القيم الفئوية والمدخلات اليدوية
# تنظيف النصوص من الفراغات المخفية وتحويل النصوص الفارغة أو الرموز إلى NaN
text_columns = df.select_dtypes(include="object").columns
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"nan": np.nan, ".......": np.nan, "": np.nan})

# توحيد مسميات الفروع المتشعبة وتصنيفها تحت تخصصات واضحة
def standardize_branch(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if any(key in val for key in ["معلوماتي", "أمن", "شبكات", "إدارية", "المعلومات", "معلوتية", "معلوطاتية", "الهندسة"]):
        return "هندسة معلوماتية"
    elif any(key in val for key in ["مدني", "مدنيه"]):
        return "هندسة مدنية"
    elif any(key in val for key in ["صيدل", "أسنان", "تمريض"]):
        return "علوم طبية وصحية"
    elif any(key in val for key in ["تربية", "معلم"]):
        return "تربية"
    return "أخرى"

df["الفرع"] = df["الفرع"].apply(standardize_branch)

# تصحيح تكرارات صياغة نوع السكن
df["السكن"] = df["السكن"].replace({"أجار": "ايجار"})

# %% [5] تصحيح أخطاء القياس والقيم الصفرية والشاذة (Outliers Cleaning)
# 1. تصحيح قيم الأصفار المصدرة بصيغ عشرية متناهية الصغر (1e-06 و 0.1) في أعمدة التأخير
df.loc[df["عدد المرات التي تأخرت بها عن تسديد القسط"] < 1, "عدد المرات التي تأخرت بها عن تسديد القسط"] = 0.0
df.loc[df["اقصى مدة زمنية تأخرت بها عن تسديد القسط بالايام"] < 1, "اقصى مدة زمنية تأخرت بها عن تسديد القسط بالايام"] = 0.0

# 2. تصحيح سلم درجات المعدل التراكمي (تحويل المدخلات بنظام 4.0 إلى سلم 100 المئوي)
gpa_col = "المعدل التراكمي التقريبي من 100"
df.loc[df[gpa_col] <= 4.0, gpa_col] *= 25.0
# اعتبار المعدلات غير المنطقية الأقل من 35% قيماً مفقودة لإعادة تعويضها
df.loc[df[gpa_col] < 35.0, gpa_col] = np.nan

# 3. عزل الساعات المسجلة الشاذة (خارج المدى الأكاديمي المسموح 3 إلى 36 ساعة)
hours_col = "عدد الساعات لمسجل عليها"
df.loc[~df[hours_col].between(3, 36), hours_col] = np.nan

# 4. عزل القسط الفصلي غير المنطقي (مدخل بقيمة 1 دولار)
tuition_col = "قيمة القسط الفصلي بشكل تقديري"
df.loc[df[tuition_col] < 50.0, tuition_col] = np.nan

# 5. عزل أعداد الإخوة غير المنطقية (مدخل بقيمة 200 أخ)
siblings_col = "عدد الاخوة والاخوات الطلاب في التعليم الجامعي الخاص"
df.loc[df[siblings_col] > 15, siblings_col] = np.nan

# 6. عزل مسافات السكن غير المنطقية الناتجة عن أخطاء كتابة (مدخل بقيمة 8000 كم)
dist_col = "بعد سكن الطالب عن مركز الجامعة بالكيلو متر"
df.loc[df[dist_col] > 300, dist_col] = np.nan

# 7. عزل عدد مرات عدم التأخير المبالغ فيها (مدخل بقيمة 876 مرة)
ontime_col = "عدد المرات التي لم تتأخر بها عن تسديد القسط"
df.loc[df[ontime_col] > 30, ontime_col] = np.nan

print(f"إجمالي الخلايا التي تم رصدها كأخطاء وتحويلها لـ NaN: {df.isna().sum().sum()}")

# %% [6] معالجة القيم المفقودة (Imputation)
# تعويض الأعمدة الرقمية باستخدام الوسيط (Median) لتجنب الانحياز
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# تعويض الأعمدة الفئوية باستخدام المنوال (Mode) الأكثر تكراراً
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

print(f"التحقق من اكتمال البيانات: عدد القيم المفقودة المتبقية = {df.isna().sum().sum()} (تمت المعالجة 100%).")

# حفظ نسخة نظيفة تماماً قبل الترميز للرجوع إليها أو عرضها
os.makedirs("data", exist_ok=True)
df.to_csv("data/data_cleaned.csv", index=False, encoding="utf-8-sig")

# %% [7] ترميز متغيرات الهدف (Target Encoding)
df_encoded = df.copy()

# هدف التصنيف (Classification): هل سدد في موعده؟ (نعم = 1، لا = 0)
target_class_col = "هل سددت القسط الأخير في موعده"
df_encoded["target_class"] = df_encoded[target_class_col].map({"نعم": 1, "لا": 0}).astype(int)

# هدف الانحدار (Regression): أقصى مدة زمنية تأخر بها بالأيام
target_delay_col = "اقصى مدة زمنية تأخرت بها عن تسديد القسط بالايام"
df_encoded["target_delay_days"] = df_encoded[target_delay_col].astype(float)

# %% [8] ترميز المتغيرات الترتيبية والثنائية (Ordinal & Binary Encoding)
# 1. ترميز السنة الدراسية كمتغير ترتيبي تصاعدي
academic_order = [["أولى", "ثانية", "ثالثة", "رابعة", "خامسة"]]
ord_year = OrdinalEncoder(categories=academic_order)
df_encoded["السنة_الدراسية_ord"] = ord_year.fit_transform(df_encoded[["السنة الدراسية"]]).astype(int)

# 2. ترميز فئات الدخل الشهري بترتيب منطقي تصاعدي
income_order = [["اقل من 100", "بين 100 و 200", "بين 200 و 500", "بين 500 و 1000", "اكثر من 1000"]]
ord_income = OrdinalEncoder(categories=income_order)
income_col = "الدخل المالي الشهري للعائلة بالدولار حصرا"
df_encoded["الدخل_الشهري_ord"] = ord_income.fit_transform(df_encoded[[income_col]]).astype(int)

# 3. ترميز الأعمدة الثنائية البسيطة (نعم = 1، لا = 0)
binary_mapping = {"نعم": 1, "لا": 0}
df_encoded["هل_تعمل_bin"] = df_encoded["هل تعمل الى جانب الدراسة"].map(binary_mapping).astype(int)
df_encoded["هل_سبق_تأخر_bin"] = df_encoded["هل سبق وأن تأخر في سداد القسط عن الموعد المحدد"].map(binary_mapping).astype(int)
df_encoded["أقساط_سابقة_غير_مسددة_bin"] = df_encoded["هل توجد أقساط جامعية سابقة غير مسددة"].map(binary_mapping).astype(int)

# حذف الأعمدة النصية الأصلية التي جرى تحويلها إلى أرقام ترتيبية وثنائية وأهداف
cols_to_drop_text = [
    "السنة الدراسية",
    "الدخل المالي الشهري للعائلة بالدولار حصرا",
    "هل تعمل الى جانب الدراسة",
    "هل سبق وأن تأخر في سداد القسط عن الموعد المحدد",
    "هل توجد أقساط جامعية سابقة غير مسددة",
    target_class_col,
    target_delay_col
]
df_encoded = df_encoded.drop(columns=cols_to_drop_text)

# %% [9] ترميز المتغيرات الاسمية بالترميز الأحادي (One-Hot Encoding)
# الأعمدة الاسمية المتبقية: الفرع، من المسؤول عن دفع القسط، المحافظة، السكن
nominal_features = ["الفرع", "من المسؤول عن دفع القسط", "المحافظة", "السكن"]
df_final = pd.get_dummies(df_encoded, columns=nominal_features, drop_first=True, dtype=int)

# %% [10] الفحص النهائي وحفظ مصفوفة البيانات المشفرة بالكامل
remaining_objects = df_final.select_dtypes(include="object").columns.tolist()
print(f"عدد الأعمدة النصية غير المشفرة المتبقية: {len(remaining_objects)}")
print(f"الأبعاد النهائية للبيانات الجاهزة للنمذجة: {df_final.shape[0]} صف و {df_final.shape[1]} عمود رقمي بالكامل.")

# حفظ الملف النهائي
df_final.to_csv("data/data_preprocessed_encoded.csv", index=False, encoding="utf-8-sig")
print("تم حفظ الملف النهائي بنجاح: 'data/data_preprocessed_encoded.csv'")