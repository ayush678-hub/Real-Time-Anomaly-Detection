import os
import json
import torch
import torch.nn.functional as F

from feature_extractor import extract_frames, CLASSES, SEVERITY_MAP, ANOMALY_MAP
from model import VideoClassifier

CHECKPOINT = os.path.join(os.path.dirname(__file__), "best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPLANATIONS = {
    "abuse":         "One or more individuals appear to be subjected to physical or psychological mistreatment, with aggressive posturing and forceful contact detected.",
    "arrest":        "Law enforcement personnel are observed restraining and detaining an individual, with controlled physical contact and authoritative body language.",
    "arson":         "Visible flames and smoke are detected alongside a person exhibiting fire-starting behavior near a structure or object.",
    "assault":       "A person is seen delivering forceful strikes or attacks against another individual who is not actively fighting back.",
    "burglary":      "An individual is observed making unauthorized entry into a building or secured area, exhibiting covert and evasive movement.",
    "explosion":     "A sudden high-energy blast with expanding fireball, debris, and shockwave is detected in the scene.",
    "fighting":      "Two or more individuals are engaged in mutual physical aggression with repeated hitting, pushing, and grappling motions.",
    "normal":        "No suspicious or anomalous activity detected. Scene shows routine human movement consistent with normal behavior.",
    "road_accident": "A collision or sudden impact between vehicles or between a vehicle and a pedestrian is detected, with abrupt motion changes.",
    "robbery":       "An individual is forcibly taking property from another person using threats or physical force.",
    "shooting":      "A person is observed wielding a firearm with shooting posture; rapid motion and potential victim reactions are detected.",
    "shoplifting":   "An individual is seen concealing or removing merchandise from a retail environment without proceeding to payment.",
    "stealing":      "A person is observed covertly taking an unattended object belonging to someone else.",
    "vandalism":     "Deliberate destruction or defacement of property is observed, including breaking, spraying, or damaging surfaces.",
}

KEY_INDICATORS = {
    "abuse":         ["forceful physical contact", "victim in distress posture", "repeated aggressive actions"],
    "arrest":        ["uniformed personnel present", "subject being restrained", "controlled takedown motion"],
    "arson":         ["visible fire or ignition", "person near flames", "smoke and rapid spread of fire"],
    "assault":       ["one-sided physical attack", "victim falling or recoiling", "aggressive striking motion"],
    "burglary":      ["unauthorized entry point", "covert movement near doors/windows", "person carrying items out"],
    "explosion":     ["sudden bright flash", "expanding debris cloud", "rapid scene disruption"],
    "fighting":      ["rapid aggressive movements", "mutual physical contact between individuals", "crowd reaction or dispersal"],
    "normal":        ["routine pedestrian movement", "no aggressive interactions", "stable scene dynamics"],
    "road_accident": ["sudden vehicle collision", "abrupt deceleration or impact", "post-crash scene disruption"],
    "robbery":       ["confrontational approach", "victim compliance under duress", "property forcibly taken"],
    "shooting":      ["firearm visible or implied", "shooting stance detected", "victim falling or fleeing"],
    "shoplifting":   ["item concealment gesture", "evasive movement near shelves", "bypassing checkout area"],
    "stealing":      ["covert object retrieval", "quick concealment motion", "unattended property taken"],
    "vandalism":     ["striking or spraying surfaces", "property damage visible", "deliberate destructive motion"],
}


def load_model():
    model = VideoClassifier(num_classes=len(CLASSES)).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    else:
        print(f"[WARNING] No checkpoint found at {CHECKPOINT}. Using untrained model.")
    model.eval()
    return model


def predict(video_path: str, model=None) -> dict:
    if model is None:
        model = load_model()

    frames = extract_frames(video_path)
    if frames is None:
        return {"error": f"Could not read video: {video_path}"}

    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    class_idx = int(probs.argmax())
    predicted_class = CLASSES[class_idx]
    confidence = round(float(probs[class_idx]) * 100, 1)

    result = {
        "predicted_class": predicted_class,
        "anomaly": ANOMALY_MAP[predicted_class],
        "severity_score": SEVERITY_MAP[predicted_class],
        "confidence": confidence,
        "explanation": EXPLANATIONS[predicted_class],
        "key_indicators": KEY_INDICATORS[predicted_class],
    }
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_video.mp4>")
        sys.exit(1)

    output = predict(sys.argv[1])
    print(json.dumps(output, indent=2))
