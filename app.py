"""AI-Based OTT Content Screening System (the "ATS for OTT content")
Run:  streamlit run app.py
"""
import os, json, time, sqlite3
import cv2, torch
import pandas as pd
import streamlit as st
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------- SETTINGS ----------
APPROVE_BELOW = 0.30          # risk under this = auto approve
HIGH_RISK = 0.70              # risk over this = flagged as HIGH priority
MODERATOR_PASSWORD = os.getenv("MOD_PASS", "admin123")
FRAME_EVERY_SEC, MAX_FRAMES = 2, 30
os.makedirs("uploads", exist_ok=True)

# Index 0 is the "safe" prompt; the rest are risky categories
CATEGORIES = {
    "Safe": "a normal safe everyday photo",
    "Nudity": "a photo of a nude person",
    "Adult/18+": "an explicit sexual adult scene",
    "Violence": "a violent scene with fighting, blood or injury",
    "Weapons": "a person holding a gun or knife",
    "Drugs/Alcohol": "people using drugs or drinking alcohol",
}

# ---------- DATABASE ----------
db = sqlite3.connect("ott.db", check_same_thread=False)
db.executescript("""
CREATE TABLE IF NOT EXISTS content(
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, creator TEXT, path TEXT,
  kind TEXT, status TEXT, risk REAL, scores TEXT, worst_sec REAL, created TEXT);
CREATE TABLE IF NOT EXISTS reviews(
  id INTEGER PRIMARY KEY AUTOINCREMENT, content_id INTEGER, reviewer TEXT,
  decision TEXT, note TEXT, time TEXT);
""")

def q(sql, args=()):
    cur = db.execute(sql, args); db.commit(); return cur

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

import altair as alt
def show_chart(scores: dict):
    """Horizontal bars, fixed order, 0-100% axis, red line at the approve limit."""
    df = pd.DataFrame({"Category": list(scores), "Risk": list(scores.values())})
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y("Category:N", sort=list(scores), title=None),
        x=alt.X("Risk:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"), title="Risk score"),
        color=alt.condition(alt.datum.Risk >= APPROVE_BELOW,
                            alt.value("#e45756"), alt.value("#54a24b")),
        tooltip=["Category", alt.Tooltip("Risk:Q", format=".0%")])
    line = alt.Chart(pd.DataFrame({"x": [APPROVE_BELOW]})).mark_rule(
        color="red", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(bars + line, use_container_width=True)

# ---------- AI MODEL (free, runs on CPU) ----------
@st.cache_resource
def load_model():
    name = "openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)

def get_frames(path, kind):
    if kind == "image":
        return [Image.open(path).convert("RGB")], [0.0]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step, i, frames, secs = int(fps * FRAME_EVERY_SEC), 0, [], []
    while len(frames) < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok: break
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        secs.append(i / fps); i += step
    cap.release()
    return frames, secs

def analyze(path, kind):
    model, proc = load_model()
    frames, secs = get_frames(path, kind)
    inputs = proc(text=list(CATEGORIES.values()), images=frames,
                  return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.softmax(dim=1)   # frames x categories
    risky = probs[:, 1:]                                          # drop "Safe"
    scores = {n: float(risky[:, j].max()) for j, n in enumerate(list(CATEGORIES)[1:])}
    risk = max(scores.values())
    worst_frame = int(risky.max(dim=1).values.argmax())
    return scores, risk, secs[worst_frame]

# ---------- PAGES ----------
st.set_page_config(page_title="OTT Content Screening", layout="wide")
st.title("🎬 AI Content Screening for OTT")
page = st.sidebar.radio("Go to", ["Creator Upload", "Moderator Panel", "Review History", "All Content"])

if page == "Creator Upload":
    st.subheader("Upload your content")
    creator = st.text_input("Creator name")
    title = st.text_input("Title")
    file = st.file_uploader("Video or image", type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png"])
    if st.button("Submit for screening") and file and creator and title:
        path = os.path.join("uploads", f"{int(time.time())}_{file.name}")
        with open(path, "wb") as f: f.write(file.getbuffer())
        kind = "image" if file.type.startswith("image") else "video"
        with st.spinner("AI is screening your content..."):
            scores, risk, worst = analyze(path, kind)
        status = "AUTO_APPROVED" if risk < APPROVE_BELOW else "FLAGGED"
        q("INSERT INTO content(title,creator,path,kind,status,risk,scores,worst_sec,created) VALUES(?,?,?,?,?,?,?,?,?)",
          (title, creator, path, kind, status, risk, json.dumps(scores), worst, now()))
        if status == "AUTO_APPROVED":
            st.success(f"✅ Approved automatically (risk {risk:.0%}). Ready to publish.")
        else:
            st.warning(f"⚠️ Sent to a human moderator (risk {risk:.0%}).")
        show_chart(scores)

elif page == "Moderator Panel":
    if st.sidebar.text_input("Moderator password", type="password") != MODERATOR_PASSWORD:
        st.info("Enter the moderator password in the sidebar."); st.stop()
    reviewer = st.sidebar.text_input("Your name", "moderator1")
    rows = q("SELECT id,title,creator,path,kind,risk,scores,worst_sec FROM content "
             "WHERE status='FLAGGED' ORDER BY risk DESC").fetchall()
    st.subheader(f"Flagged queue ({len(rows)})")
    for cid, title, creator, path, kind, risk, scores, worst in rows:
        tag = "🔴 HIGH" if risk >= HIGH_RISK else "🟠 MEDIUM"
        with st.expander(f"{tag} | {title} by {creator} | risk {risk:.0%}"):
            c1, c2 = st.columns(2)
            with c1:
                if kind == "image": st.image(path)
                else:
                    st.video(path); st.caption(f"Most risky moment: around {worst:.0f}s")
            with c2:
                show_chart(json.loads(scores))
                note = st.text_area("Reason / note", key=f"n{cid}")
                a, r = st.columns(2)
                for col, label, new in [(a, "✅ Approve", "APPROVED"), (r, "❌ Reject", "REJECTED")]:
                    if col.button(label, key=f"{new}{cid}"):
                        q("UPDATE content SET status=? WHERE id=?", (new, cid))
                        q("INSERT INTO reviews(content_id,reviewer,decision,note,time) VALUES(?,?,?,?,?)",
                          (cid, reviewer, new, note, now()))
                        st.rerun()

elif page == "Review History":
    st.subheader("Moderator decisions")
    df = pd.read_sql("SELECT r.time, c.title, c.creator, c.risk, r.decision, r.reviewer, r.note "
                     "FROM reviews r JOIN content c ON c.id=r.content_id ORDER BY r.id DESC", db)
    st.dataframe(df, use_container_width=True)

else:
    st.subheader("All content and status")
    df = pd.read_sql("SELECT id,title,creator,kind,status,round(risk,2) AS risk,created "
                     "FROM content ORDER BY id DESC", db)
    st.dataframe(df, use_container_width=True)