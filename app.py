"""
============================================================
 EmojiMatch API — Flask Backend for Render Deployment
 Compatible with ALL 5 trained models.

 Notification format sent to ALL users:
   Title : {emoji} {course title}   →  "📚 Intro to ML"
   Body  : {subtitle}               →  "Week 1 material uploaded"
============================================================
"""
from flask import Flask, request, jsonify
import os
import json
import joblib
import random
import firebase_admin
from firebase_admin import credentials, messaging
from datetime import datetime, timezone

app = Flask(__name__)

# ── ML Model ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAMES = [
    "emoji_tfidf_sgd.joblib",
    "emoji_tfidf_lr.joblib",
    "emoji_tfidf_svc.joblib",
    "emoji_tfidf_mnb.joblib",
    "emoji_tfidf_cnb.joblib",
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
    raise FileNotFoundError("No model .joblib found! Upload one of: " + ", ".join(MODEL_FILENAMES))

# ── Emoji Map ─────────────────────────────────────────────────────────────────
EMOJI_FILE    = os.path.join(BASE_DIR, "cat_emoji_map.json")
CAT_EMOJI_MAP = {}

if os.path.exists(EMOJI_FILE):
    with open(EMOJI_FILE, "r", encoding="utf-8") as f:
        CAT_EMOJI_MAP = json.load(f)
    print(f"✅ Loaded emoji map: {len(CAT_EMOJI_MAP)} categories")
else:
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
    print("⚠️ cat_emoji_map.json not found — using built-in fallback")


# ── Firebase lazy init ────────────────────────────────────────────────────────
def init_firebase():
    if firebase_admin._apps:
        return
    firebase_key_json = os.environ.get("FIREBASE_KEY_JSON", "")
    if not firebase_key_json:
        raise RuntimeError(
            "FIREBASE_KEY_JSON is not set. "
            "Render Dashboard → Environment → Add Variable."
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
        "status":     "ok",
        "service":    "emoji-notification-api",
        "model":      MODEL_NAME,
        "categories": list(CAT_EMOJI_MAP.keys()),
    }), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME}), 200


@app.post("/predict")
def predict():
    """
    Input:  { "title": "Your assignment is due tomorrow" }
    Output: { "label": "Assignment Reminder", "emoji": "🚨",
              "top3_emojis": ["🚨","🔔","⏰"], "confidence": 0.94,
              "notif_title": "🚨 Your assignment is due tomorrow" }
    """
    data = request.get_json(silent=True) or {}
    text = (
        request.form.get("title")
        or request.form.get("tittle")
        or data.get("title")
        or data.get("tittle")
        or data.get("text")
        or ""
    ).strip()

    if not text:
        return jsonify({"error": "Provide 'title' or 'text' field"}), 400

    label     = model.predict([text])[0]
    emojis    = CAT_EMOJI_MAP.get(label, ["📌", "📢", "🔔"])
    top_emoji = random.choice(emojis)  # pick randomly from top 3

    confidence = None
    try:
        confidence = round(float(model.predict_proba([text]).max()), 4)
    except Exception:
        pass

    response = {
        "label":       label,
        "emoji":       top_emoji,
        "top3_emojis": emojis,
        "notif_title": f"{top_emoji} {text}",
    }
    if confidence is not None:
        response["confidence"] = confidence

    return jsonify(response), 200


@app.post("/send-notification")
def send_notification():
    """
    Called by Android after saving a course to Firebase.
    Predicts emoji from title → sends FCM to topic "all_users".

    Notification shown on EVERY subscribed device:
      ┌─────────────────────────────────┐
      │  📚 Introduction to ML          │  ← emoji + title
      │  Week 1 material uploaded       │  ← subtitle
      └─────────────────────────────────┘

    Input JSON:
      {
        "courseId":  "abc123",
        "title":     "Introduction to Machine Learning",
        "subtitle":  "Week 1 material uploaded",
        "imageUrl":  "https://...",
        "pdfLink":   "https://...",
        "predictedClass": "Study Material"   (optional)
      }
    """
    try:
        init_firebase()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Firebase init failed: {e}"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    # ── Read fields ───────────────────────────────────────────
    course_id = str(data.get("courseId",  ""))
    title     = str(data.get("title",     "New Course Added")).strip()
    subtitle  = str(data.get("subtitle",  "")).strip()
    image_url = str(data.get("imageUrl",  ""))
    pdf_link  = str(data.get("pdfLink",   ""))

    # ── Timestamp — generated ONCE on the server ──────────────
    # All users receive the SAME timestamp so every device shows
    # the exact same "sent at" time regardless of when they open it.
    now        = datetime.now(timezone.utc)
    sent_at_ts = str(int(now.timestamp() * 1000))   # Unix ms  e.g. "1714900800000"
    sent_at_fmt = now.strftime("%d %b %Y, %I:%M %p UTC")  # e.g. "25 Apr 2025, 03:30 PM UTC"

    # ── Predict emoji from title ──────────────────────────────
    predicted_class = str(data.get("predictedClass", "")).strip()
    predicted_emoji = str(data.get("predictedEmoji", "")).strip()

    if not predicted_class and title:
        predicted_class = model.predict([title])[0]

    if not predicted_emoji and predicted_class:
        predicted_emoji = random.choice(CAT_EMOJI_MAP.get(predicted_class, ["📌", "📢", "🔔"]))  # pick randomly

    # ── Notification strings ──────────────────────────────────
    #    Title → emoji + course title    "📚 Introduction to ML"
    #    Body  → subtitle                "Week 1 material uploaded"
    notif_title = f"{predicted_emoji} {title}"
    notif_body  = subtitle if subtitle else title

    print(f"📤 Sending FCM → all_users | '{notif_title}' | '{notif_body}'")

    # ── FCM message — DATA ONLY (no notification block) ─────────────────────
    #
    # WHY data-only:
    #   If you include a "notification" block, Android OS handles the
    #   notification itself when the app is CLOSED — it always launches
    #   the LAUNCHER activity (Splash_screen) ignoring our PendingIntent.
    #
    #   With data-only, FCM always calls onMessageReceived() on the device
    #   and our service builds the notification with the correct PendingIntent
    #   pointing to CourseDetailActivity — works whether app is open,
    #   background, or fully closed.
    #
    message = messaging.Message(

        # ALL data fields — Android reads in onMessageReceived()
        data={
            "courseId":       course_id,
            "title":          title,
            "subtitle":       subtitle,
            "imageUrl":       image_url,
            "pdfLink":        pdf_link,
            "predictedClass": predicted_class,
            "predictedEmoji": predicted_emoji,
            "notif_title":    notif_title,
            "notif_body":     notif_body,
            "sentAt":         sent_at_ts,     # Unix ms  — same for all devices
            "sentAtFormatted": sent_at_fmt,   # human-readable — same for all devices
            "emojiEnabled":    "true",    # flag so Android knows emoji is on
        },

        # NO notification block — our Android service handles display
        # so PendingIntent (→ CourseDetailActivity) is always respected

        android=messaging.AndroidConfig(
            priority="high",   # wake device even in Doze mode
        ),

        topic="all_users",
    )

    try:
        fcm_response = messaging.send(message)
        print(f"✅ FCM sent: {fcm_response}")
        return jsonify({
            "success":        True,
            "fcm_id":         fcm_response,
            "notif_title":    notif_title,
            "notif_body":     notif_body,
            "predictedClass": predicted_class,
            "predictedEmoji": predicted_emoji,
            "sentAt":         sent_at_ts,
            "sentAtFormatted": sent_at_fmt,
        }), 200

    except Exception as e:
        print(f"❌ FCM error: {e}")
        return jsonify({"error": str(e)}), 500



@app.post("/send-notification-plain")
def send_notification_plain():
    """
    Sends notification WITHOUT emoji — plain text only.
    Used when emoji toggle is OFF in AdminPlanActivity.

    Notification format:
      Title : {course title}           (no emoji)
      Body  : {subtitle}
    """
    try:
        init_firebase()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Firebase init failed: {e}"}), 500

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    course_id = str(data.get("courseId",  ""))
    title     = str(data.get("title",     "New Course Added")).strip()
    subtitle  = str(data.get("subtitle",  "")).strip()
    image_url = str(data.get("imageUrl",  ""))
    pdf_link  = str(data.get("pdfLink",   ""))

    now          = datetime.now(timezone.utc)
    sent_at_ts   = str(int(now.timestamp() * 1000))
    sent_at_fmt  = now.strftime("%d %b %Y, %I:%M %p UTC")

    # Plain — no emoji at all
    notif_title = title
    notif_body  = subtitle if subtitle else title

    print(f"📤 Sending PLAIN FCM → all_users | '{notif_title}' | '{notif_body}'")

    message = messaging.Message(
        data={
            "courseId":        course_id,
            "title":           title,
            "subtitle":        subtitle,
            "imageUrl":        image_url,
            "pdfLink":         pdf_link,
            "predictedClass":  "",
            "predictedEmoji":  "",
            "notif_title":     notif_title,
            "notif_body":      notif_body,
            "sentAt":          sent_at_ts,
            "sentAtFormatted": sent_at_fmt,
            "emojiEnabled":    "false",   # flag so Android knows it's plain
        },
        android=messaging.AndroidConfig(priority="high"),
        topic="all_users",
    )

    try:
        fcm_response = messaging.send(message)
        print(f"✅ PLAIN FCM sent: {fcm_response}")
        return jsonify({
            "success":        True,
            "fcm_id":         fcm_response,
            "notif_title":    notif_title,
            "notif_body":     notif_body,
            "emojiEnabled":   False,
            "sentAt":         sent_at_ts,
            "sentAtFormatted": sent_at_fmt,
        }), 200
    except Exception as e:
        print(f"❌ FCM error: {e}")
        return jsonify({"error": str(e)}), 500


@app.get("/emoji-map")
def emoji_map():
    return jsonify(CAT_EMOJI_MAP), 200


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# """
# ============================================================
#  EmojiMatch API — Flask Backend for Render Deployment
#  Compatible with ALL 5 trained models.
#
#  Notification format sent to ALL users:
#    Title : {emoji} {course title}   →  "📚 Intro to ML"
#    Body  : {subtitle}               →  "Week 1 material uploaded"
# ============================================================
# """
# from flask import Flask, request, jsonify
# import os
# import json
# import joblib
# import firebase_admin
# from firebase_admin import credentials, messaging
#
# app = Flask(__name__)
#
# # ── ML Model ──────────────────────────────────────────────────────────────────
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#
# MODEL_FILENAMES = [
#     "emoji_tfidf_sgd.joblib",
#     "emoji_tfidf_lr.joblib",
#     "emoji_tfidf_svc.joblib",
#     "emoji_tfidf_mnb.joblib",
#     "emoji_tfidf_cnb.joblib",
# ]
#
# model      = None
# MODEL_NAME = None
#
# for fname in MODEL_FILENAMES:
#     path = os.path.join(BASE_DIR, fname)
#     if os.path.exists(path):
#         model      = joblib.load(path)
#         MODEL_NAME = fname
#         print(f"✅ Loaded model: {fname}")
#         break
#
# if model is None:
#     raise FileNotFoundError("No model .joblib found! Upload one of: " + ", ".join(MODEL_FILENAMES))
#
# # ── Emoji Map ─────────────────────────────────────────────────────────────────
# EMOJI_FILE    = os.path.join(BASE_DIR, "cat_emoji_map.json")
# CAT_EMOJI_MAP = {}
#
# if os.path.exists(EMOJI_FILE):
#     with open(EMOJI_FILE, "r", encoding="utf-8") as f:
#         CAT_EMOJI_MAP = json.load(f)
#     print(f"✅ Loaded emoji map: {len(CAT_EMOJI_MAP)} categories")
# else:
#     CAT_EMOJI_MAP = {
#         "Assignment Status":    ["😊", "😀", "🎉"],
#         "Performance Feedback": ["💪", "🙌", "🥳"],
#         "Study Material":       ["📚", "📌", "📢"],
#         "Attendance":           ["😞", "📊", "📝"],
#         "Class Schedule":       ["⏰", "❌", "⏳"],
#         "Assignment Feedback":  ["💬", "📝", "📋"],
#         "Announcement":         ["📢", "📌", "📚"],
#         "Assignment Alert":     ["😢", "😞", "⚠️"],
#         "Assignment Reminder":  ["🚨", "🔔", "⏰"],
#         "Exam Schedule":        ["📅", "📢", "🚨"],
#         "Security Alert":       ["⚠️", "🔒", "❗"],
#         "Feedback Reminder":    ["⏳", "🙂", "😅"],
#     }
#     print("⚠️ cat_emoji_map.json not found — using built-in fallback")
#
#
# # ── Firebase lazy init ────────────────────────────────────────────────────────
# def init_firebase():
#     if firebase_admin._apps:
#         return
#     firebase_key_json = os.environ.get("FIREBASE_KEY_JSON", "")
#     if not firebase_key_json:
#         raise RuntimeError(
#             "FIREBASE_KEY_JSON is not set. "
#             "Render Dashboard → Environment → Add Variable."
#         )
#     key_dict = json.loads(firebase_key_json)
#     cred     = credentials.Certificate(key_dict)
#     firebase_admin.initialize_app(cred)
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# # ROUTES
# # ─────────────────────────────────────────────────────────────────────────────
#
# @app.get("/")
# def root():
#     return jsonify({
#         "status":     "ok",
#         "service":    "emoji-notification-api",
#         "model":      MODEL_NAME,
#         "categories": list(CAT_EMOJI_MAP.keys()),
#     }), 200
#
#
# @app.get("/health")
# def health():
#     return jsonify({"status": "ok", "model": MODEL_NAME}), 200
#
#
# @app.post("/predict")
# def predict():
#     """
#     Input:  { "title": "Your assignment is due tomorrow" }
#     Output: { "label": "Assignment Reminder", "emoji": "🚨",
#               "top3_emojis": ["🚨","🔔","⏰"], "confidence": 0.94,
#               "notif_title": "🚨 Your assignment is due tomorrow" }
#     """
#     data = request.get_json(silent=True) or {}
#     text = (
#         request.form.get("title")
#         or request.form.get("tittle")
#         or data.get("title")
#         or data.get("tittle")
#         or data.get("text")
#         or ""
#     ).strip()
#
#     if not text:
#         return jsonify({"error": "Provide 'title' or 'text' field"}), 400
#
#     label     = model.predict([text])[0]
#     emojis    = CAT_EMOJI_MAP.get(label, ["📌", "📢", "🔔"])
#     top_emoji = emojis[0]
#
#     confidence = None
#     try:
#         confidence = round(float(model.predict_proba([text]).max()), 4)
#     except Exception:
#         pass
#
#     response = {
#         "label":       label,
#         "emoji":       top_emoji,
#         "top3_emojis": emojis,
#         "notif_title": f"{top_emoji} {text}",
#     }
#     if confidence is not None:
#         response["confidence"] = confidence
#
#     return jsonify(response), 200
#
#
# @app.post("/send-notification")
# def send_notification():
#     """
#     Called by Android after saving a course to Firebase.
#     Predicts emoji from title → sends FCM to topic "all_users".
#
#     Notification shown on EVERY subscribed device:
#       ┌─────────────────────────────────┐
#       │  📚 Introduction to ML          │  ← emoji + title
#       │  Week 1 material uploaded       │  ← subtitle
#       └─────────────────────────────────┘
#
#     Input JSON:
#       {
#         "courseId":  "abc123",
#         "title":     "Introduction to Machine Learning",
#         "subtitle":  "Week 1 material uploaded",
#         "imageUrl":  "https://...",
#         "pdfLink":   "https://...",
#         "predictedClass": "Study Material"   (optional)
#       }
#     """
#     try:
#         init_firebase()
#     except RuntimeError as e:
#         return jsonify({"error": str(e)}), 500
#     except Exception as e:
#         return jsonify({"error": f"Firebase init failed: {e}"}), 500
#
#     data = request.get_json(silent=True)
#     if not data:
#         return jsonify({"error": "No JSON body received"}), 400
#
#     # ── Read fields ───────────────────────────────────────────
#     course_id = str(data.get("courseId",  ""))
#     title     = str(data.get("title",     "New Course Added")).strip()
#     subtitle  = str(data.get("subtitle",  "")).strip()
#     image_url = str(data.get("imageUrl",  ""))
#     pdf_link  = str(data.get("pdfLink",   ""))
#
#     # ── Predict emoji from title ──────────────────────────────
#     predicted_class = str(data.get("predictedClass", "")).strip()
#     predicted_emoji = str(data.get("predictedEmoji", "")).strip()
#
#     if not predicted_class and title:
#         predicted_class = model.predict([title])[0]
#
#     if not predicted_emoji and predicted_class:
#         predicted_emoji = CAT_EMOJI_MAP.get(predicted_class, ["📌"])[0]
#
#     # ── Notification strings ──────────────────────────────────
#     #    Title → emoji + course title    "📚 Introduction to ML"
#     #    Body  → subtitle                "Week 1 material uploaded"
#     notif_title = f"{predicted_emoji} {title}"
#     notif_body  = subtitle if subtitle else title
#
#     print(f"📤 Sending FCM → all_users | '{notif_title}' | '{notif_body}'")
#
#     # ── FCM message ───────────────────────────────────────────
#     message = messaging.Message(
#
#         # data payload — readable in onMessageReceived() in Android
#         data={
#             "courseId":       course_id,
#             "title":          title,
#             "subtitle":       subtitle,
#             "imageUrl":       image_url,
#             "pdfLink":        pdf_link,
#             "predictedClass": predicted_class,
#             "predictedEmoji": predicted_emoji,
#         },
#
#         # notification payload — OS renders this in the system tray
#         notification=messaging.Notification(
#             title=notif_title,   # "📚 Introduction to ML"
#             body=notif_body,     # "Week 1 material uploaded"
#         ),
#
#         android=messaging.AndroidConfig(
#             priority="high",
#             notification=messaging.AndroidNotification(
#                 sound="default",
#                 channel_id="course_add_channel",
#                 priority="high",
#             ),
#         ),
#
#         # ✅ ALL devices subscribed to "all_users" receive this
#         topic="all_users",
#     )
#
#     try:
#         fcm_response = messaging.send(message)
#         print(f"✅ FCM sent: {fcm_response}")
#         return jsonify({
#             "success":        True,
#             "fcm_id":         fcm_response,
#             "notif_title":    notif_title,
#             "notif_body":     notif_body,
#             "predictedClass": predicted_class,
#             "predictedEmoji": predicted_emoji,
#         }), 200
#
#     except Exception as e:
#         print(f"❌ FCM error: {e}")
#         return jsonify({"error": str(e)}), 500
#
#
# @app.get("/emoji-map")
# def emoji_map():
#     return jsonify(CAT_EMOJI_MAP), 200
#
#
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)
