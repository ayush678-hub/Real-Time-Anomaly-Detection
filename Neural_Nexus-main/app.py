import json
import gradio as gr
from predict import predict, load_model


def severity_icon(score):
    if score <= 2:  return "🟢 Normal"
    if score <= 5:  return "🟡 Mild"
    if score <= 8:  return "🟠 Serious"
    return "🔴 Critical"


model = load_model()


def analyze(video_path):
    if video_path is None:
        return "⚠️ Please upload a video file.", ""

    result = predict(video_path, model=model)
    if "error" in result:
        return f"❌ {result['error']}", ""

    icon  = severity_icon(result["severity_score"])
    summary = (
        f"### 🔍 Detection Result\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| **Class** | `{result['predicted_class'].upper()}` |\n"
        f"| **Anomaly** | {result['anomaly']} |\n"
        f"| **Severity** | {icon} — {result['severity_score']} / 10 |\n"
        f"| **Confidence** | {result['confidence']}% |\n\n"
        f"**📝 Explanation:**  \n{result['explanation']}\n\n"
        f"**🔑 Key Indicators:**\n" +
        "\n".join(f"- {k}" for k in result["key_indicators"])
    )
    return summary, json.dumps(result, indent=2)


with gr.Blocks(title="AI Video Surveillance Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 AI-Powered Video Surveillance Analysis")
    gr.Markdown(
        "Upload a CCTV video clip. The system will classify the activity, "
        "detect anomalies, and assign a severity score (1–10)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input  = gr.Video(label="📹 Upload Video Clip")
            analyze_btn  = gr.Button("🔎 Analyze", variant="primary")
        with gr.Column(scale=2):
            summary_out  = gr.Markdown(label="Analysis Summary")

    with gr.Accordion("📄 Raw JSON Output", open=False):
        json_out = gr.Code(label="JSON", language="json")

    analyze_btn.click(
        fn=analyze,
        inputs=video_input,
        outputs=[summary_out, json_out]
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
