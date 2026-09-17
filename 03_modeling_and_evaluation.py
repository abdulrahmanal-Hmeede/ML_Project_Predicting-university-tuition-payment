# %% [1] استيراد المكتبات وحزم التعلم الآلي
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# أدوات التقسيم والتحجيم واختيار الميزات
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# نماذج التصنيف والانحدار
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import GaussianNB

# مقاييس التقييم
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error
)

# ضبط نمط الرسوم البيانية
sns.set_theme(style="whitegrid")

# %% [2] تحميل البيانات المشفرة وفصل الميزات عن الأهداف
# قراءة ملف البيانات الرقمية بالكامل الناتج عن مرحلة المعالجة السابقة
data_file = "data/data_preprocessed_encoded.csv"
if not os.path.exists(data_file):
    data_file = "data_preprocessed_encoded.csv"

df = pd.read_csv(data_file)
print(f"أبعاد مصفوفة البيانات الرقمية: {df.shape[0]} صف و {df.shape[1]} عمود.")

# عزل مصفوفة الميزات المستقلة X عن هدفي المشروع
X = df.drop(columns=["target_class", "target_delay_days"])
y_class = df["target_class"]         # هدف التصنيف (1: سدد بالموعد، 0: تأخر)
y_delay = df["target_delay_days"]     # هدف الانحدار (عدد أيام التأخير)

print(f"عدد الميزات الإجمالي المدخلة للنمذجة: {X.shape[1]}")
print(f"توزيع فئات هدف التصنيف: {dict(y_class.value_counts())}")

# %% [3] تقسيم البيانات والتحجيم الإحصائي
# تقسيم البيانات إلى 80% تدريب و 20% اختبار مع التوزيع الطبقي لضمان تمثيل الفئتين
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.20, random_state=42, stratify=y_class
)

print(f"عينات التدريب: {X_train.shape[0]} | عينات الاختبار: {X_test.shape[0]}")

# تحجيم الميزات باستخدام StandardScaler للخوارزميات الحسابية والمسافية
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [4] اختيار الميزات وتحليل كسب المعلومات (Feature Selection)
# قياس كسب المعلومات (Mutual Information) لعزل الميزات الأكثر تأثيراً على السداد
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
feature_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\n--- أهم 10 عوامل مؤثرة في قرار سداد القسط (Mutual Information) ---")
print(feature_importance.head(10))

# رسم بياني لأهم الميزات وحفظه للملف البحثي
plt.figure(figsize=(10, 5))
feature_importance.head(10).plot(kind="barh", color="#1f77b4")
plt.title("أهم العوامل المؤثرة على سداد القسط بموعده (Mutual Information)", fontsize=13)
plt.xlabel("درجة كسب المعلومات (Information Gain)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_features_importance.png", dpi=300)
plt.show()

# %% [5] تدريب ومقارنة نماذج التصنيف (7 خوارزميات)
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Nearest Centroid": NearestCentroid()
}

results = []

for name, model in classification_models.items():
    # النماذج المعتمدة على المسافات تستخدم البيانات المقيسة، والأشجار تستخدم الأصلية
    if name in ["Logistic Regression", "SVM", "KNN", "Nearest Centroid"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    results.append({
        "Algorithm": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    })

results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\n--- جدول المقارنة الشامل لنماذج التصنيف ---")
print(results_df.to_string(index=False))

# %% [6] الرسم البياني لمقارنة أداء النماذج
results_df.set_index("Algorithm")[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(12, 6)
)
plt.title("مقارنة خوارزميات التصنيف في التنبؤ بسداد القسط", fontsize=14)
plt.ylabel("الدرجة (0 إلى 1)")
plt.ylim(0, 1.1)
plt.xticks(rotation=30)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("classification_models_comparison.png", dpi=300)
plt.show()

# %% [7] مصفوفة الارتباك للنموذج الأفضل (Random Forest)
best_model_name = results_df.iloc[0]["Algorithm"]
best_model = classification_models[best_model_name]
print(f"\nالنموذج المتصدر حسب F1 Score هو: {best_model_name}")

if best_model_name in ["Logistic Regression", "SVM", "KNN", "Nearest Centroid"]:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

print("\n--- تقرير التصنيف التفصيلي لأفضل نموذج ---")
print(classification_report(y_test, y_pred_best, target_names=["تأخر (0)", "سدد بموعده (1)"]))

# رسم مصفوفة الارتباك
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["متأخر (0)", "سدد بموعده (1)"],
            yticklabels=["متأخر (0)", "سدد بموعده (1)"])
plt.title(f"مصفوفة الارتباك (Confusion Matrix) - {best_model_name}", fontsize=12)
plt.xlabel("التنبؤ (Predicted)")
plt.ylabel("القيمة الفعلية (Actual)")
plt.tight_layout()
plt.savefig("confusion_matrix_best_model.png", dpi=300)
plt.show()

# %% [8] التحقق المتقاطع الطبقي للنموذج (Stratified K-Fold CV)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring="f1")

print("\n--- تقييم ثبات النموذج عبر التحقق المتقاطع (5-Fold CV) ---")
for fold_no, score in enumerate(cv_scores, start=1):
    print(f"المرحلة {fold_no}: F1 = {score:.4f}")
print(f"المتوسط العام لمقياس F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %% [9] تدريب نموذج الانحدار للتنبؤ بعدد أيام التأخير
# تدريب نموذج الانحدار على الطلاب الذين لديهم تأخير فعلي (> 0)
delayed_mask = y_delay > 0
X_reg = X[delayed_mask]
y_reg = y_delay[delayed_mask]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_r, y_train_r)

reg_preds = reg_model.predict(X_test_r)
mae = mean_absolute_error(y_test_r, reg_preds)
rmse = np.sqrt(mean_squared_error(y_test_r, reg_preds))

print("\n--- نتائج نموذج الانحدار (التنبؤ بعدد أيام التأخير) ---")
print(f"متوسط الخطأ المطلق (MAE): {mae:.2f} يوم")
print(f"جذر متوسط مربعات الخطأ (RMSE): {rmse:.2f} يوم")

# %% [10] تصدير وحفظ النماذج عبر Pickle
os.makedirs("models", exist_ok=True)

# 1. حفظ نموذج التصنيف الفائز
with open("models/classifier_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# 2. حفظ نموذج الانحدار
with open("models/regressor_model.pkl", "wb") as f:
    pickle.dump(reg_model, f)

# 3. حفظ أداة التحجيم الإحصائي
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 4. حفظ ترتيب وأسماء الأعمدة للربط مع API
with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nتم تصدير وحفظ جميع الملفات في مجلد 'models/' بنجاح تام.")