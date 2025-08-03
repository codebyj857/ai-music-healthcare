import streamlit as st
import webbrowser
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dummy data for demonstration — replace with your actual balanced dataset
X_balanced = np.random.rand(1000, 2)
y_balanced = np.random.randint(0, 8, 1000)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Emotion to music mapping
emotion_to_music = {
    0: "Calm instrumental",
    1: "Upbeat pop",
    2: "Energetic workout tracks",
    3: "Focus-enhancing ambient",
    4: "Happy dance tunes",
    6: "Relaxing nature sounds",
    7: "Soothing acoustic"
}

# Streamlit UI
st.title("🎧 Emotion-Based Music Recommender")

feature1 = st.slider("Feature 1", 0.0, 1.0, 0.5)
feature2 = st.slider("Feature 2", 0.0, 1.0, 0.5)
sample = np.array([feature1, feature2]).reshape(1, -1)

if st.button("Recommend"):
    emotion = model.predict(sample)[0]
    music = emotion_to_music.get(emotion, "Default playlist")
    st.success(f"Predicted Emotion: {emotion}")
    st.info(f"Recommended Music: {music}")
    url = f"https://www.youtube.com/results?search_query={music} playlist"
    webbrowser.open(url)
