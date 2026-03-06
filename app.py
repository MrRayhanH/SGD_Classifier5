"""
============================================================
 EmojiMatch API — Flask Backend for Render Deployment
 Compatible with ALL 5 trained models:
   emoji_tfidf_mnb.joblib  (Model 1 - Multinomial NB)
   emoji_tfidf_cnb.joblib  (Model 2 - Complement NB)
   emoji_tfidf_svc.joblib  (Model 3 - Linear SVC)
   emoji_tfidf_sgd.joblib  (Model 4 - SGD ★ Recommended)
   emoji_tfidf_lr.joblib   (Model 5 - Logistic Regression)

 Upload whichever .joblib you trained on Kaggle. SGD_Classifier5
 Also upload cat_emoji_map.json alongside it.
============================================================
"""
from flask import Flask, request, jsonify
import os
import json
import joblib
import firebase_admin
from firebase_admin import credentials, messaging

app = Flask(__name__)

# ── ML Model ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Auto-detect which model file is present (tries SGD first as recommended)
MODEL_FILENAMES = [
    "emoji_tfidf_sgd.joblib",   # Model 4 — SGD (★ recommended)
    "emoji_tfidf_lr.joblib",    # Model 5 — Logistic Regression
    "emoji_tfidf_svc.joblib",   # Model 3 — Linear SVC
    "emoji_tfidf_mnb.joblib",   # Model 1 — Multinomial NB
    "emoji_tfidf_cnb.joblib",   # Model 2 — Complement NB
]

model      = None
MODEL_NAME = None

for fname in MODEL_FILENAMES:
    path = os.path.join(BASE_DIR, fname)
    if os.path.exists(path):
        model      = joblib.load(path)
        MODEL_NAME = fname
        print(f"✅ Loaded model: {fname}")
        break

if model is None:
    raise FileNotFoundError(
        "No model .joblib found! Upload one of: " + ", ".join(MODEL_FILENAMES)
    )

# ── Emoji Map ─────────────────────────────────────────────────────────────────
EMOJI_FILE = os.path.join(BASE_DIR, "cat_emoji_map.json")
CAT_EMOJI_MAP = {}

if os.path.exists(EMOJI_FILE):
    with open(EMOJI_FILE, "r", encoding="utf-8") as f:
        CAT_EMOJI_MAP = json.load(f)
    print(f"✅ Loaded emoji map: {len(CAT_EMOJI_MAP)} categories")
else:
    # Fallback hard-coded map if file is missing
    CAT_EMOJI_MAP = {
        "Assignment Status":    ["😊", "😀", "🎉"],
        "Performance Feedback": ["💪", "🙌", "🥳"],
        "Study Material":       ["📚", "📌", "📢"],
        "Attendance":           ["😞", "📊", "📝"],
        "Class Schedule":       ["⏰", "❌", "⏳"],
        "Assignment Feedback":  ["💬", "📝", "📋"],
        "Announcement":         ["📢", "📌", "📚"],
        "Assignment Alert":     ["😢", "😞", "⚠️"],
        "Assignment Reminder":  ["🚨", "🔔", "⏰"],
        "Exam Schedule":        ["📅", "📢", "🚨"],
        "Security Alert":       ["⚠️", "🔒", "❗"],
        "Feedback Reminder":    ["⏳", "🙂", "😅"],
    }
    print("⚠️ cat_emoji_map.json not found — using fallback emoji map")

# ── Firebase: lazy init ───────────────────────────────────────────────────────
def init_firebase():
    if firebase_admin._apps:
        return
    firebase_key_json = os.environ.get("FIREBASE_KEY_JSON", "")
    if not firebase_key_json:
        raise RuntimeError(
            "FIREBASE_KEY_JSON is not set. "
            "Go to Render Dashboard → Environment → Add Variable."
        )
    key_dict = json.loads(firebase_key_json)
    cred     = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return jsonify({
        "status":  "ok",
        "service": "emoji-notification-api",
        "model":   MODEL_NAME,
        "categories": list(CAT_EMOJI_MAP.keys()),
    }), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME}), 200


@app.post("/predict")
def predict():
    """
    Input (JSON or form):
        { "title": "Your assignment is due tomorrow" }
        or  { "text": "..." }

    Response:
        {
            "label":       "Assignment Reminder",
            "emoji":       "🚨",
            "top3_emojis": ["🚨", "🔔", "⏰"],
            "confidence":  0.94
        }
    """
    data = request.get_json(silent=True) or {}
    text = (
        request.form.get("title")
        or request.form.get("tittle")      # keep backward-compat typo
        or data.get("title")
        or data.get("tittle")
        or data.get("text")
        or ""
    ).strip()

    if not text:
        return jsonify({"error": "Provide 'title' or 'text' field"}), 400

    # Predict category
    label = model.predict([text])[0]

    # Confidence (available for SGD, LR, SVC-calibrated, NB models)
    confidence = None
    try:
        proba      = model.predict_proba([text])
        confidence = round(float(proba.max()), 4)
    except Exception:
        pass  # LinearSVC without calibration won't have proba

    # Get emojis for this category
    emojis     = CAT_EMOJI_MAP.get(label, ["📌", "📢", "🔔"])
    top_emoji  = emojis[0]

    response = {
        "label":       label,
        "emoji":       top_emoji,
        "top3_emojis": emojis,
    }
    if confidence is not None:
        response["confidence"] = confidence

    return jsonify(response), 200


@app.post("/predict-batch")
def predict_batch():
    """
    Input: { "texts": ["sentence 1", "sentence 2", ...] }
    Response: list of predictions, one per text
    Useful for predicting multiple notifications at once.
    """
    data  = request.get_json(silent=True) or {}
    texts = data.get("texts", [])

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Provide 'texts' as a JSON array"}), 400

    if len(texts) > 50:
        return jsonify({"error": "Max 50 texts per batch"}), 400

    results = []
    labels  = model.predict(texts)

    try:
        probas = model.predict_proba(texts).max(axis=1)
    except Exception:
        probas = [None] * len(texts)

    for i, (text, label) in enumerate(zip(texts, labels)):
        emojis = CAT_EMOJI_MAP.get(label, ["📌", "📢", "🔔"])
        entry  = {
            "text":        text,
            "label":       label,
            "emoji":       emojis[0],
            "top3_emojis": emojis,
        }
        if probas[i] is not None:
            entry["confidence"] = round(float(probas[i]), 4)
        results.append(entry)

    return jsonify({"predictions": results, "count": len(results)}), 200


@app.post("/send-notification")
def send_notification():
    """
    Called by Android InsertCourseActivity after a new course is saved.
    Automatically predicts emoji from course title.
    Sends FCM push to ALL devices subscribed to topic 'all_users'.

    Input:
        {
            "courseId":       "abc123",
            "title":          "Introduction to Machine Learning",
            "subtitle":       "Week 1 material uploaded",
            "imageUrl":       "https://...",
            "pdfLink":        "https://...",
            "predictedClass": "Study Material"   ← optional, auto-predicted if missing
        }
    """
    try:
        init_firebase()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Firebase init failed: " + str(e)}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    course_id    = str(data.get("courseId",  ""))
    title        = str(data.get("title",     "New Course Added"))
    subtitle     = str(data.get("subtitle",  ""))
    image_url    = str(data.get("imageUrl",  ""))
    pdf_link     = str(data.get("pdfLink",   ""))

    # Auto-predict category + emoji if not provided
    predicted_class = str(data.get("predictedClass", "")).strip()
    predicted_emoji = str(data.get("predictedEmoji", "")).strip()

    if not predicted_class and title:
        predicted_class = model.predict([title])[0]

    if not predicted_emoji and predicted_class:
        emojis          = CAT_EMOJI_MAP.get(predicted_class, ["📌"])
        predicted_emoji = emojis[0]

    # Build FCM message
    notification_title = f"{predicted_emoji} New Course Added!"
    notification_body  = title if title else "Check the latest update"

    message = messaging.Message(
        data={
            "courseId":       course_id,
            "title":          title,
            "subtitle":       subtitle,
            "imageUrl":       image_url,
            "pdfLink":        pdf_link,
            "predictedClass": predicted_class,
            "predictedEmoji": predicted_emoji,
        },
        notification=messaging.Notification(
            title=notification_title,
            body=notification_body,
        ),
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                sound="default",
                channel_id="course_add_channel",
            ),
        ),
        topic="all_users",
    )

    try:
        response = messaging.send(message)
        print(f"✅ FCM sent: {response} | emoji={predicted_emoji} | class={predicted_class}")
        return jsonify({
            "success":        True,
            "fcm_id":         response,
            "predictedClass": predicted_class,
            "predictedEmoji": predicted_emoji,
        }), 200
    except Exception as e:
        print(f"❌ FCM error: {e}")
        return jsonify({"error": str(e)}), 500


@app.get("/emoji-map")
def emoji_map():
    """Returns the full category → emoji mapping. Useful for Android app reference."""
    return jsonify(CAT_EMOJI_MAP), 200


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
