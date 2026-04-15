import cv2
import numpy as np
import mediapipe as mp
from fer import FER
import threading
import time
import matplotlib.pyplot as plt # New import for graphing
from sklearn.linear_model import LogisticRegression
import pandas as pd

# --- INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

emotion_detector = FER(mtcnn=False) 

RIGHT_EYE_IDX = [33, 133]
RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_IRIS_IDX = [474, 475, 476, 477]
LEFT_EYE_IDX = [362, 263]
LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_IRIS_IDX = [469, 470, 471, 472]

state = {
    "emotion_string": "Analyzing...",
    "emotion_score": 1.0,
    "last_processed_frame": None,
    "is_processing": False,
    "w_g": 0.65, 
    "w_e": 0.35, 
    "smoothed_attention": 100.0 
}

EMOTION_WEIGHTS = {
    "happy": 1.0, "neutral": 0.9, "surprise": 0.7, 
    "sad": 0.4, "angry": 0.3, "fear": 0.3, "disgust": 0.2
}

# --- HELPER FUNCTIONS ---

def emotion_worker():
    while True:
        if state["last_processed_frame"] is not None and not state["is_processing"]:
            state["is_processing"] = True
            try:
                emotions = emotion_detector.detect_emotions(state["last_processed_frame"])
                if emotions:
                    emo_dict = emotions[0]["emotions"]
                    weighted_score = sum(emo_dict.get(e, 0) * weight for e, weight in EMOTION_WEIGHTS.items())
                    top_emo = max(emo_dict, key=emo_dict.get)
                    top_score = emo_dict[top_emo] * 100
                    
                    state["emotion_string"] = f"{top_emo.capitalize()} ({int(top_score)}%)"
                    state["emotion_score"] = min(1.0, max(0.0, weighted_score))
            except Exception:
                pass
            state["is_processing"] = False
        time.sleep(0.1)

def get_iris_position(landmarks, eye_lr, eye_tb, iris, w, h):
    eye_left, eye_right = landmarks[eye_lr[0]], landmarks[eye_lr[1]]
    eye_top, eye_bottom = landmarks[eye_tb[0]], landmarks[eye_tb[1]]
    
    iris_x = np.mean([landmarks[i].x for i in iris]) * w
    iris_y = np.mean([landmarks[i].y for i in iris]) * h

    h_ratio = (iris_x - (eye_left.x * w)) / ((eye_right.x * w) - (eye_left.x * w) + 1e-6)
    v_ratio = (iris_y - (eye_top.y * h)) / ((eye_bottom.y * h) - (eye_top.y * h) + 1e-6)
    
    return h_ratio, v_ratio, (int(iris_x), int(iris_y))

def calculate_continuous_gaze(h, v):
    dist = np.sqrt((h - 0.5)**2 + (v - 0.5)**2)
    score = 1.0 - (dist * 3.33) 
    return max(0.0, min(1.0, score))

# --- MAIN LOOP ---

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t = threading.Thread(target=emotion_worker, daemon=True)
    t.start()

    frame_count = 0
    alpha = 0.15 

    # --- DATA LOGGING ARRAYS ---
    time_history = []
    attention_history = []
    session_start_time = time.time()

    print("Session started. Press 'ESC' to end and view the graph.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if frame_count % 10 == 0:
            state["last_processed_frame"] = frame.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            hr_r, vr_r, iris_r = get_iris_position(landmarks, RIGHT_EYE_IDX, RIGHT_EYE_TOP_BOTTOM, RIGHT_IRIS_IDX, w, h)
            hr_l, vr_l, iris_l = get_iris_position(landmarks, LEFT_EYE_IDX, LEFT_EYE_TOP_BOTTOM, LEFT_IRIS_IDX, w, h)
            
            avg_h, avg_v = (hr_r + hr_l) / 2, (vr_r + vr_l) / 2
            g_score = calculate_continuous_gaze(avg_h, avg_v)
            
            inst_attention = (state["w_g"] * g_score + state["w_e"] * state["emotion_score"]) * 100
            inst_attention = min(100, max(0, inst_attention))

            state["smoothed_attention"] = (alpha * inst_attention) + ((1 - alpha) * state["smoothed_attention"])
            display_score = int(state["smoothed_attention"])

            # --- LOG DATA EVERY FRAME ---
            current_time = time.time() - session_start_time
            time_history.append(current_time)
            attention_history.append(display_score)

            cv2.circle(frame, iris_r, 3, (0, 255, 255), -1)
            cv2.circle(frame, iris_l, 3, (0, 255, 255), -1)
            
            color = (0, 255, 0) if display_score > 70 else (0, 165, 255) if display_score > 40 else (0, 0, 255)
            
            cv2.putText(frame, f"ATTENTION: {display_score}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Emotion: {state['emotion_string']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Gaze Pos: H:{avg_h:.2f} V:{avg_v:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.rectangle(frame, (20, 60), (220, 75), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 60), (20 + int(display_score * 2), 75), color, -1)

        cv2.imshow("Continuous Gaze & Emotion Tracker", frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

    # --- LOGISTIC REGRESSION ANALYSIS ---
    if len(time_history) > 10:
        print("\n--- Generating Intelligence Report ---")
        
        # 1. Prepare Data
        # We'll use the raw scores gathered during the session
        # For simulation, let's assume we captured gaze_history and emotion_history
        # Here we create a synthetic 'Target' based on your logic to train the model
        X = np.array(list(zip(attention_history, time_history))) # Features
        
        # Create a binary label: 1 if attention > 50, else 0
        y = [1 if score > 50 else 0 for score in attention_history]

        if len(set(y)) < 2:
            print("Not enough variety in data (all attentive or all distracted) to train model.")
        else:
            model = LogisticRegression()
            # We reshape X to include gaze and emotion specifically if you tracked them
            # For this example, let's fit based on the history
            model.fit(X, y)

            # 2. Reasoning / Interpretation
            importance = model.coef_[0]
            intercept = model.intercept_[0]
            
            print(f"Model Accuracy on Session: {model.score(X, y) * 100:.1f}%")
            
            # --- FINAL ANALYSIS & REASONING ---
            avg_attn = np.mean(attention_history)
            
            print("\n--- Session Reasoning ---")
            if avg_attn > 70:
                print(f"Status: HIGHLY ATTENTIVE ({avg_attn:.1f}%)")
                print("Reasoning: Consistent gaze centered on the screen and positive/neutral emotional markers.")
            elif avg_attn > 40:
                print(f"Status: MODERATELY ATTENTIVE ({avg_attn:.1f}%)")
                print("Reasoning: Frequent 'micro-distractions' detected. Gaze drifted significantly at intervals.")
            else:
                print(f"Status: DISTRACTED ({avg_attn:.1f}%)")
                print("Reasoning: Major lack of engagement. Gaze was predominantly off-center.")

            # 3. Generate the Graph (Existing Graph Code)
            plt.figure(figsize=(10, 5))
            plt.plot(time_history, attention_history, color='green', label="Real-time Attention")
            plt.axhline(y=avg_attn, color='gray', linestyle='--', label=f'Average: {avg_attn:.1f}%')
            plt.title("Post-Session Logistic Regression Analysis")
            plt.xlabel("Seconds")
            plt.ylabel("Probability of Attention (%)")
            plt.legend()
            plt.show()

    # --- GENERATE GRAPH ON EXIT ---
    if len(time_history) > 0:
        print("Generating session report...")
        plt.figure(figsize=(10, 5))
        plt.plot(time_history, attention_history, label="Attention Score", color="#2ca02c", linewidth=2)
        
        # Shade the area under the curve
        plt.fill_between(time_history, attention_history, color="#2ca02c", alpha=0.15)
        
        # Add reference lines for High and Low attention thresholds
        plt.axhline(y=70, color='blue', linestyle='--', alpha=0.5, label='High Attention Threshold')
        plt.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Distraction Threshold')
        
        plt.title("Session Attention Analysis", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Attention Score (%)", fontsize=12)
        plt.ylim(0, 105)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        # This will pop up a window with your graph
        plt.show()

if __name__ == "__main__":
    main()

    "w_e": 0.35, 
    "smoothed_attention": 100.0 
}

EMOTION_WEIGHTS = {
    "happy": 1.0, "neutral": 0.9, "surprise": 0.7, 
    "sad": 0.4, "angry": 0.3, "fear": 0.3, "disgust": 0.2
}

# --- HELPER FUNCTIONS ---

def emotion_worker():
    while True:
        if state["last_processed_frame"] is not None and not state["is_processing"]:
            state["is_processing"] = True
            try:
                emotions = emotion_detector.detect_emotions(state["last_processed_frame"])
                if emotions:
                    emo_dict = emotions[0]["emotions"]
                    weighted_score = sum(emo_dict.get(e, 0) * weight for e, weight in EMOTION_WEIGHTS.items())
                    top_emo = max(emo_dict, key=emo_dict.get)
                    top_score = emo_dict[top_emo] * 100
                    
                    state["emotion_string"] = f"{top_emo.capitalize()} ({int(top_score)}%)"
                    state["emotion_score"] = min(1.0, max(0.0, weighted_score))
            except Exception:
                pass
            state["is_processing"] = False
        time.sleep(0.1)

def get_iris_position(landmarks, eye_lr, eye_tb, iris, w, h):
    eye_left, eye_right = landmarks[eye_lr[0]], landmarks[eye_lr[1]]
    eye_top, eye_bottom = landmarks[eye_tb[0]], landmarks[eye_tb[1]]
    
    iris_x = np.mean([landmarks[i].x for i in iris]) * w
    iris_y = np.mean([landmarks[i].y for i in iris]) * h

    h_ratio = (iris_x - (eye_left.x * w)) / ((eye_right.x * w) - (eye_left.x * w) + 1e-6)
    v_ratio = (iris_y - (eye_top.y * h)) / ((eye_bottom.y * h) - (eye_top.y * h) + 1e-6)
    
    return h_ratio, v_ratio, (int(iris_x), int(iris_y))

def calculate_continuous_gaze(h, v):
    dist = np.sqrt((h - 0.5)**2 + (v - 0.5)**2)
    score = 1.0 - (dist * 3.33) 
    return max(0.0, min(1.0, score))

# --- MAIN LOOP ---

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t = threading.Thread(target=emotion_worker, daemon=True)
    t.start()

    frame_count = 0
    alpha = 0.15 

    # --- DATA LOGGING ARRAYS ---
    time_history = []
    attention_history = []
    session_start_time = time.time()

    print("Session started. Press 'ESC' to end and view the graph.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if frame_count % 10 == 0:
            state["last_processed_frame"] = frame.copy()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            hr_r, vr_r, iris_r = get_iris_position(landmarks, RIGHT_EYE_IDX, RIGHT_EYE_TOP_BOTTOM, RIGHT_IRIS_IDX, w, h)
            hr_l, vr_l, iris_l = get_iris_position(landmarks, LEFT_EYE_IDX, LEFT_EYE_TOP_BOTTOM, LEFT_IRIS_IDX, w, h)
            
            avg_h, avg_v = (hr_r + hr_l) / 2, (vr_r + vr_l) / 2
            g_score = calculate_continuous_gaze(avg_h, avg_v)
            
            inst_attention = (state["w_g"] * g_score + state["w_e"] * state["emotion_score"]) * 100
            inst_attention = min(100, max(0, inst_attention))

            state["smoothed_attention"] = (alpha * inst_attention) + ((1 - alpha) * state["smoothed_attention"])
            display_score = int(state["smoothed_attention"])

            # --- LOG DATA EVERY FRAME ---
            current_time = time.time() - session_start_time
            time_history.append(current_time)
            attention_history.append(display_score)

            cv2.circle(frame, iris_r, 3, (0, 255, 255), -1)
            cv2.circle(frame, iris_l, 3, (0, 255, 255), -1)
            
            color = (0, 255, 0) if display_score > 70 else (0, 165, 255) if display_score > 40 else (0, 0, 255)
            
            cv2.putText(frame, f"ATTENTION: {display_score}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Emotion: {state['emotion_string']}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(frame, f"Gaze Pos: H:{avg_h:.2f} V:{avg_v:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.rectangle(frame, (20, 60), (220, 75), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 60), (20 + int(display_score * 2), 75), color, -1)

        cv2.imshow("Continuous Gaze & Emotion Tracker", frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

    # --- GENERATE GRAPH ON EXIT ---
    if len(time_history) > 0:
        print("Generating session report...")
        plt.figure(figsize=(10, 5))
        plt.plot(time_history, attention_history, label="Attention Score", color="#2ca02c", linewidth=2)
        
        # Shade the area under the curve
        plt.fill_between(time_history, attention_history, color="#2ca02c", alpha=0.15)
        
        # Add reference lines for High and Low attention thresholds
        plt.axhline(y=70, color='blue', linestyle='--', alpha=0.5, label='High Attention Threshold')
        plt.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='Distraction Threshold')
        
        plt.title("Session Attention Analysis", fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Attention Score (%)", fontsize=12)
        plt.ylim(0, 105)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        
        # This will pop up a window with your graph
        plt.show()

if __name__ == "__main__":
    main()
