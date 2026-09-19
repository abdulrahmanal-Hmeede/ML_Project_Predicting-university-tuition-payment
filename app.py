import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 1. تهيئة تطبيق FastAPI ومحرك القوالب
app = FastAPI(
    title="Tuition Payment Prediction System",
    description="نظام ذكي للتنبؤ بسداد القسط الجامعي وتقدير أيام التأخير",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 2. مسارات ملفات النماذج المحفوظة عبر pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier_model.pkl")
REGRESSOR_PATH = os.path.join(MODELS_DIR, "regressor_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

# 3. تحميل النماذج والمحولات إلى الذاكرة
with open(CLASSIFIER_PATH, "rb") as f:
  classifier_model = pickle.load(f)

with open(REGRESSOR_PATH, "rb") as f:
  regressor_model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
  scaler = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
  feature_columns = pickle.load(f)


# 4. الصفحة الرئيسية: عرض واجهة الاستمارة
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
  return templates.TemplateResponse(
      "index.html", {"request": request, "result": None}
  )


# 5. نقطة النهاية لمعالجة التنبؤ واستقبال بيانات النموذج
@app.post("/predict", response_class=HTMLResponse)
async def predict_tuition(
    request: Request,
    hours: float = Form(...),
    gpa: float = Form(...),
    attendance: float = Form(...),
    delayed_times: float = Form(...),
    not_delayed_times: float = Form(...),
    siblings: float = Form(...),
    tuition: float = Form(...),
    workers: float = Form(...),
    distance: float = Form(...),
    academic_year: str = Form(...),
    monthly_income: str = Form(...),
    is_working: str = Form(...),
    past_delay: str = Form(...),
    unpaid_installments: str = Form(...),
    faculty: str = Form(...),
    payer: str = Form(...),
    province: str = Form(...),
    housing: str = Form(...),
):
  # إنشاء مصفوفة ميزات مصفّرة تطابق هيكل التدريب تماماً
  features_dict = {col: 0.0 for col in feature_columns}

  # تعيين القيم الرقمية
  features_dict["عدد الساعات لمسجل عليها"] = float(hours)
  features_dict["المعدل التراكمي التقريبي من 100"] = float(gpa)
  features_dict["نسبة الحضور في جميع المقررات من 100"] = float(attendance)
  features_dict["عدد المرات التي تأخرت بها عن تسديد القسط"] = float(
      delayed_times
  )
  features_dict["عدد المرات التي لم تتأخر بها عن تسديد القسط"] = float(
      not_delayed_times
  )
  features_dict["عدد الاخوة والاخوات الطلاب في التعليم الجامعي الخاص"] = float(
      siblings
  )
  features_dict["قيمة القسط الفصلي بشكل تقديري"] = float(tuition)
  features_dict["عدد الاشخاص العاملين بالاسرة"] = float(workers)
  features_dict["بعد سكن الطالب عن مركز الجامعة بالكيلو متر"] = float(distance)

  # ترميز المتغيرات الترتيبية (Ordinal Encoding)
  academic_order = {"أولى": 0, "ثانية": 1, "ثالثة": 2, "رابعة": 3, "خامسة": 4}
  features_dict["السنة_الدراسية_ord"] = float(
      academic_order.get(academic_year, 2)
  )

  income_order = {
      "اقل من 100": 0,
      "بين 100 و 200": 1,
      "بين 200 و 500": 2,
      "بين 500 و 1000": 3,
      "اكثر من 1000": 4,
  }
  features_dict["الدخل_الشهري_ord"] = float(
      income_order.get(monthly_income, 2)
  )

  # ترميز المتغيرات الثنائية (Binary Encoding)
  features_dict["هل_تعمل_bin"] = 1.0 if is_working == "نعم" else 0.0
  features_dict["هل_سبق_تأخر_bin"] = 1.0 if past_delay == "نعم" else 0.0
  features_dict["أقساط_سابقة_غير_مسددة_bin"] = (
      1.0 if unpaid_installments == "نعم" else 0.0
  )

  # تفعيل أعمدة الترميز الأحادي (One-Hot Encoding) إذا وُجدت
  faculty_col = f"الفرع_{faculty}"
  if faculty_col in features_dict:
    features_dict[faculty_col] = 1.0

  payer_col = f"من المسؤول عن دفع القسط_{payer}"
  if payer_col in features_dict:
    features_dict[payer_col] = 1.0

  province_col = f"المحافظة_{province}"
  if province_col in features_dict:
    features_dict[province_col] = 1.0

  housing_col = f"السكن_{housing}"
  if housing_col in features_dict:
    features_dict[housing_col] = 1.0

  # تحويل القاموس إلى DataFrame بنفس ترتيب أعمدة التدريب
  input_df = pd.DataFrame([features_dict], columns=feature_columns)

  # 1. التنبؤ بالتصنيف (احتمالية السداد)
  prediction_class = classifier_model.predict(input_df)[0]
  probabilities = classifier_model.predict_proba(input_df)[0]
  prob_on_time = round(probabilities[1] * 100, 1)

  # 2. التنبؤ بالانحدار (أيام التأخير المتوقعة)
  predicted_delay_days = None
  if prediction_class == 0:  # إذا كان متوقعاً أن يتأخر
    pred_days = regressor_model.predict(input_df)[0]
    predicted_delay_days = max(1, int(round(pred_days)))

  result_payload = {
      "on_time": bool(prediction_class == 1),
      "probability": prob_on_time,
      "delay_days": predicted_delay_days if predicted_delay_days is not None else 0,
  }

  return templates.TemplateResponse(
      "index.html", {"request": request, "result": result_payload}
  )