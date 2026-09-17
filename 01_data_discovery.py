# %% [1] استيراد المكتبات وضبط بيئة العرض
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ضبط النمط البياني الافتراضي (خلفية بيضاء مع خطوط شبكية واضحة)
sns.set_theme(style="whitegrid")

# %% [2] تحميل البيانات الأولية ومعاينة الأبعاد
# قراءة ملف الإكسل عبر محرك openpyxl المدمج
df = pd.read_excel("data/مجموعة البيانات.xlsx")

print("--- أبعاد مجموعة البيانات ---")
print(f"إجمالي عدد الطلاب (الصفوف): {df.shape[0]}")
print(f"إجمالي عدد المتغيرات (الأعمدة): {df.shape[1]}")
print("-" * 50)

# عرض أول صفين لمعاينة بنية البيانات
df.head(2)

# %% [3] فحص أسماء الأعمدة وأنواع البيانات البرمجية
print("--- قائمة الأعمدة لكشف الفراغات الخفية ---")
for index, col in enumerate(df.columns, start=1):
  # دالة repr تكشف أي مسافات زائدة في أسماء الأعمدة
  print(f"{index}. {repr(col)}")

print("\n--- فحص أنواع البيانات (Dtypes) والقيم غير الفارغة ---")
df.info()

# %% [4] تقرير القيم المفقودة (Missing Values Report)
# حساب عدد ونسبة الفراغات في كل عمود
missing_counts = df.isnull().sum()
missing_ratios = (df.isnull().mean() * 100).round(2)

missing_df = pd.DataFrame(
    {"عدد_القيم_المفقودة": missing_counts, "النسبة_%": missing_ratios}
)

# تصفية الأعمدة التي تحوي مفقودات فقط وترتيبها تنازلياً
missing_summary = missing_df[missing_df["عدد_القيم_المفقودة"] > 0].sort_values(
    by="عدد_القيم_المفقودة", ascending=False
)

print("--- تفاصيل الأعمدة التي تحتوي على قيم مفقودة ---")
print(missing_summary)
print(f"\nإجمالي عدد الخلايا الفارغة في الجدول: {df.isna().sum().sum()}")

# %% [5] فحص التكرارات التامة في السجلات
# التحقق مما إذا قام طالب بإرسال النموذج مرتين بالخطأ
duplicate_count = df.duplicated().sum()
print(f"عدد السجلات المكررة بالكامل: {duplicate_count}")

# %% [6] الإحصاء الوصفي للأعمدة الرقمية وكشف الشواذ (Outliers)
num_cols = df.select_dtypes(include=[np.number]).columns

print("--- الإحصاءات الوصفية للمتغيرات العددية ---")
# استخراج المقاييس الأساسية لرصد القيم الشاذة والمستحيلة
df[num_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

# %% [7] فحص المتغيرات الفئوية (Categorical) وتشتت الصياغات
cat_cols = df.select_dtypes(include=["object"]).columns

print("--- تكرارات القيم في المتغيرات الفئوية (كشف الأخطاء الإملائية) ---")
for col in cat_cols:
  print(f"\n--- المتغير: {col} ---")
  print(df[col].value_counts(dropna=False).head(6))

# %% [8] فحص توازن فئات المتغير المستهدف (Target Class)
# عمود التصنيف: هل تم سداد القسط الأخير بموعده
target_col = "  هل سددت القسط الأخير في موعده  "

target_freq = df[target_col].value_counts(dropna=False)
target_pct = (
    df[target_col].value_counts(normalize=True, dropna=False) * 100
).round(2)

target_report = pd.DataFrame(
    {"العدد": target_freq, "النسبة_المئوية_%": target_pct}
)
print("--- توازن فئات متغير الهدف ---")
print(target_report)

# %% [9] الرسوم البيانية الاستكشافية للمشاكل المكتشفة
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. مخطط توزيع متغير الهدف
sns.countplot(data=df, x=target_col, ax=axes[0, 0], palette="Blues")
axes[0, 0].set_title("توزيع فئات سداد القسط في الموعد (عدم توازن الفئات)")
axes[0, 0].set_xlabel("هل سددت القسط في موعده؟")
axes[0, 0].set_ylabel("عدد الطلاب")

# 2. كشف الشواذ في الساعات المسجلة
sns.boxplot(
    data=df, x="عدد الساعات لمسجل عليها", ax=axes[0, 1], color="#e74c3c"
)
axes[0, 1].set_title("القيم الشاذة في الساعات المسجلة (تصل إلى 117 ساعة)")
axes[0, 1].set_xlabel("عدد الساعات")

# 3. كشف خلل سلم درجات المعدل التراكمي
sns.histplot(
    data=df,
    x="المعدل التراكمي التقريبي من 100",
    kde=True,
    ax=axes[1, 0],
    color="#8e44ad",
)
axes[1, 0].set_title("توزيع المعدل التراكمي (مدخلات بنظام 4 بدلاً من 100)")
axes[1, 0].set_xlabel("المعدل")
axes[1, 0].set_ylabel("التكرار")

# 4. كشف الشواذ في مسافة السكن
sns.boxplot(
    data=df,
    x="بعد سكن الطالب عن مركز الجامعة بالكيلو متر",
    ax=axes[1, 1],
    color="#3498db",
)
axes[1, 1].set_title("القيم الشاذة في بعد السكن (تصل إلى 8000 كم)")
axes[1, 1].set_xlabel("المسافة بالكيلومتر")

plt.tight_layout()
plt.savefig("data_discovery_report.png", dpi=300)
plt.show()