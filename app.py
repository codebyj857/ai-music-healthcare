import streamlit as st
import webbrowser
import numpy as np
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 🚀 Page config
st.set_page_config(
    page_title="Emotion-Aware Music 🎶",
    page_icon="🎧",
    layout="centered"
)

# 🌈 Final Glowing Gradient & Font Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');

    .stApp {
        background: linear-gradient(135deg, #2b1e4f, #2a2a72, #553c9a, #8e44ad, #c44569, #2c3e50);
        background-size: 600% 600%;
        animation: gradientBG 30s ease infinite;
        color: #ffffff;
        font-family: 'Lobster', cursive;
    }

    @keyframes gradientBG {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }

    h1, h2, h4 {
        text-shadow: 0 0 12px #ff00ff, 0 0 20px #ff00ff;
    }

    .slider-label {
        font-size: 20px;
        color: #ffea00;
        text-shadow: 0 0 10px #ffea00;
        font-weight: bold;
    }

    .result-box {
        background-color: rgba(255, 255, 255, 0.15);
        color: #ffffff;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.4);
        margin-top: 30px;
    }

    .stButton>button {
        background-color: #ff00ff;
        color: #fff;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 0 0 10px #ff00ff;
        border: none;
    }

    .stButton>button:hover {
        background-color: #ff66cc;
        box-shadow: 0 0 15px #ff66cc;
        transform: scale(1.05);
    }

    .stRadio > div {
        justify-content: center;
        gap: 20px;
        color: #ffffff !important;
    }

    .stSlider > div {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🎧 Intro
st.markdown("<h1>🎧 Welcome to Your Emotion-Aware Music Recommender!</h1>", unsafe_allow_html=True)
st.markdown("<h4>Let’s match your heartbeat with a beat that heals 💗</h4>", unsafe_allow_html=True)

# 📊 Sliders
st.markdown('<p class="slider-label">🔋 Energy Level</p>', unsafe_allow_html=True)
energy_level = st.slider("⚡", 0.0, 1.0, 0.5, key="energy_slider")

st.markdown('<p class="slider-label">😰 Stress Level</p>', unsafe_allow_html=True)
stress_level = st.slider("😓", 0.0, 1.0, 0.5, key="stress_slider")

st.markdown('<p class="slider-label">❤️ Heart Rate (bpm)</p>', unsafe_allow_html=True)
heart_rate = st.slider("💓", 50, 150, 75, key="heart_slider")

# 🧠 Emotion model
X_balanced = np.random.rand(1000, 3)
y_balanced = np.random.randint(0, 8, 1000)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

emotion_to_music = {
    0: "Calm instrumental 🎻",
    1: "Upbeat pop 🎤",
    2: "Energetic workout tracks 🏋️",
    3: "Focus-enhancing ambient 🎧",
    4: "Happy dance tunes 💃",
    6: "Relaxing nature sounds 🍃",
    7: "Soothing acoustic 🎸"
}

# 🎯 Prediction
normalized_hr = (heart_rate - 50) / 100
sample = np.array([energy_level, stress_level, normalized_hr]).reshape(1, -1)

# 💾 History tracker
if 'mood_history' not in st.session_state:
    st.session_state.mood_history = []

# 🎵 Button & logic
if st.button("🔮 Recommend Music"):
    with st.spinner("✨ Syncing with your vibe..."):
        time.sleep(2)
        proba = model.predict_proba(sample)[0]
        emotion = np.argmax(proba)
        confidence = round(proba[emotion] * 100, 2)
        music = emotion_to_music.get(emotion, "Default playlist 🎼")

        st.session_state.mood_history.append({
            "Energy": energy_level,
            "Stress": stress_level,
            "HeartRate": heart_rate,
            "Emotion": emotion,
        })

    # 🎶 Display results
    st.markdown(f"""
        <div class="result-box">
            <h2>🧠 Emotion: {emotion}</h2>
            <h2>🎵 Music Match: {music}</h2>
            <p>💫 Confidence Level: {confidence}%</p>
        </div>
    """, unsafe_allow_html=True)

    webbrowser.open(f"https://www.youtube.com/results?search_query={music}")

    # 📈 Mood graph
    history = st.session_state.mood_history
    if history:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(history))),
            y=[h["Emotion"] for h in history],
            mode='lines+markers',
            marker=dict(color='cyan', size=10),
            line=dict(color='magenta', width=3),
            name="Your Emotions 🎨"
        ))
        fig.update_layout(
            title="🌟 Your Mood Over Time",
            xaxis_title="Session 💬",
            yaxis_title="Emotion 🎭",
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

# 🌈 Feedback
st.markdown("#### 🎯 Did we hit the vibe right?")
st.radio("Your vibe check:", ["👍 Loved it!", "👎 Not quite there"], horizontal=True)

# 🌸 Footer
st.markdown("##### 💖 Made with magic by Joya | Music that feels you 🎶")
