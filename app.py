"""
Gradio app — Car Damage Detection & Cost Estimator
Pipeline: YOLOv8 Detection + Groq Vision LLM + Rule-based Cost Estimator
Run: python app.py
"""

import os
import glob
import base64
import io
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
from groq import Groq
from pathlib import Path
from cost_estimator import estimate

try:
    import cv2
except ImportError:
    cv2 = None

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# YOLO model — lazy loaded
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = Path(__file__).parent / "model" / "best.pt"

_yolo_model = None
_yolo_loaded = False


def get_yolo_model():
    """Load YOLO model once. Returns None if unavailable."""
    global _yolo_model, _yolo_loaded
    if _yolo_loaded:
        return _yolo_model
    _yolo_loaded = True
    if not YOLO_WEIGHTS.exists():
        return None
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO(str(YOLO_WEIGHTS))
    except Exception:
        _yolo_model = None
    return _yolo_model


# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "broken-glass": (0, 0, 255),     # red (BGR)
    "deformation":  (0, 165, 255),   # orange
    "rust":         (0, 255, 255),   # yellow
    "scratch":      (255, 0, 0),     # blue
}


def run_yolo(pil_image: Image.Image):
    """
    Run YOLO detection on image.
    Returns (detections_list, annotated_np_rgb) or (None, None) if model unavailable.
    """
    model = get_yolo_model()
    if model is None or cv2 is None:
        return None, None

    np_image = np.array(pil_image)
    results = model(np_image, conf=0.50, iou=0.5, verbose=False)

    if not results or len(results[0].boxes) == 0:
        return [], np_image

    detections = []
    annotated_bgr = cv2.cvtColor(np_image.copy(), cv2.COLOR_RGB2BGR)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        detections.append({
            "label": label,
            "confidence": conf,
            "box": (int(x1), int(y1), int(x2), int(y2)),
        })

        color = CLASS_COLORS.get(label, (0, 255, 0))
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated_bgr, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return detections, annotated_rgb


# ---------------------------------------------------------------------------
# Format YOLO results as context for LLM
# ---------------------------------------------------------------------------
def format_yolo_context(detections, cost_result):
    """Format YOLO detections + cost estimate as text for the LLM prompt."""
    if not detections:
        return "YOLO Detection: No damage objects detected by the model."

    lines = ["YOLO Detection Results:"]
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["box"]
        lines.append(
            f"  {i}. {det['label']} (confidence: {det['confidence']:.0%}) "
            f"at box [{x1},{y1},{x2},{y2}]"
        )

    lines.append(f"\nRule-based Cost Estimate:")
    for item in cost_result["items"]:
        lines.append(
            f"  - {item.label.replace('-', ' ').title()}: {item.severity}, "
            f"Rs {item.cost_low:,} - Rs {item.cost_high:,}"
        )
    lines.append(
        f"  Total: Rs {cost_result['total_low']:,} - Rs {cost_result['total_high']:,}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Indian automobile damage assessor and repair cost estimator.
You work for a leading Indian insurance company. When given a car damage image, you must:

1. **Identify the vehicle type** (hatchback, sedan, SUV, luxury, etc.)
2. **Detect ALL visible damage** — dents, scratches, cracks, broken glass, rust, paint damage, structural damage, bumper damage, etc.
3. **For each damage, specify:**
   - Damaged part (e.g., front bumper, rear door, hood, fender, headlight, windshield, etc.)
   - Damage type (scratch, dent, crack, shatter, rust, paint peel, deformation, etc.)
   - Severity (Minor / Moderate / Severe / Critical)
4. **Estimate repair cost in Indian Rupees (INR)** based on the Indian market.

## Indian Market Repair Cost Reference (2024-2025):

### Labour Rates (Indian workshops):
- Local garage / roadside: Rs 300-800/hr
- Authorised service center: Rs 800-2,000/hr
- Premium / luxury brand service center: Rs 2,000-5,000/hr

### Common Repair Costs (Non-luxury hatchback/sedan like Maruti, Hyundai, Tata):
- **Minor scratch (single panel):** Rs 1,500 - Rs 4,000
- **Deep scratch (single panel):** Rs 3,000 - Rs 8,000
- **Small dent (paintless dent repair):** Rs 1,000 - Rs 3,000
- **Medium dent + repaint:** Rs 4,000 - Rs 10,000
- **Large dent / panel beating + paint:** Rs 8,000 - Rs 20,000
- **Bumper repair (minor):** Rs 2,000 - Rs 5,000
- **Bumper replacement (front/rear):** Rs 5,000 - Rs 15,000
- **Headlight/taillight replacement:** Rs 2,000 - Rs 12,000
- **Windshield replacement:** Rs 5,000 - Rs 15,000
- **Side mirror replacement:** Rs 1,500 - Rs 6,000
- **Full panel replacement (door/fender):** Rs 15,000 - Rs 40,000
- **Full body paint (single panel):** Rs 3,000 - Rs 8,000
- **Alloy wheel repair:** Rs 1,500 - Rs 4,000

### Premium / Luxury vehicles (Honda, Toyota, BMW, Mercedes, Audi):
- Multiply above costs by 1.5x to 4x depending on brand
- OEM parts can cost 2x-5x more than local/aftermarket parts

### Structural Damage:
- **Chassis straightening:** Rs 15,000 - Rs 50,000
- **Frame damage repair:** Rs 30,000 - Rs 1,00,000+
- **Suspension repair:** Rs 5,000 - Rs 25,000
- **Radiator replacement:** Rs 3,000 - Rs 15,000

## Output Format:
Use this EXACT markdown format:

### Vehicle Assessment
**Vehicle Type:** [detected type]

### Damage Detection

| # | Damaged Part | Damage Type | Severity | Estimated Cost (INR) |
|---|-------------|-------------|----------|---------------------|
| 1 | [part] | [type] | [severity] | Rs X,XXX - Rs X,XXX |
| 2 | ... | ... | ... | ... |

### Repair Recommendation
[Brief recommendation: repair vs replace, local vs authorized service center]

### Total Estimated Cost
**Rs X,XXX - Rs X,XXX**

*(Costs are approximate based on Indian market rates for local/aftermarket parts.
Authorised service center costs may be 1.5x-2x higher.)*

## Important Rules:
- If the image is NOT a car or vehicle, say "This does not appear to be a vehicle image."
- If no damage is visible, say "No visible damage detected. Vehicle appears to be in good condition."
- Always give a cost RANGE (min-max), never a single number.
- Be specific about which part of the car is damaged.
- Consider paint work is almost always needed with dent/scratch repairs.
- Factor in Indian market conditions — local garages are much cheaper than authorized centers.

## When YOLO Detection Data Is Provided:
You may receive pre-computed YOLO object detection results along with the image.
- Use these detections as a STARTING POINT — they tell you what an object detection model found.
- VALIDATE each detection by looking at the image. If a detection seems like a false positive, say so and exclude it from costs.
- ADD any damage you see that YOLO missed — the model only detects 4 classes (broken-glass, deformation, rust, scratch) and may miss other types of damage.
- The rule-based cost estimate provided is a rough baseline. Refine it using your judgment of vehicle type, part location, and severity.
- Your final cost table should be your OWN assessment informed by (but not blindly copying) the YOLO data.
"""


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------
def analyze_damage(image: Image.Image, yolo_context: str = "") -> str:
    """Send image + optional YOLO context to Groq and get damage assessment."""
    if not API_KEY:
        return (
            "## API Key Missing!\n\n"
            "Please set your Groq API key:\n\n"
            "```\nset GROQ_API_KEY=your_key_here\n```\n\n"
            "Then restart the app. Get a free key at: "
            "https://console.groq.com/keys"
        )

    client = Groq(api_key=API_KEY)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    user_text = (
        "Analyze this car damage image and provide a detailed assessment "
        "with Indian market repair costs."
    )
    if yolo_context:
        user_text += (
            "\n\nThe following YOLO object detection results were pre-computed "
            "for this image. Use them as context but validate against what you see:\n\n"
            + yolo_context
        )

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return (
                "## Rate Limit\n\n"
                "Groq rate limit hit. Please wait a few seconds and try again.\n"
                "Free tier allows ~30 requests/minute."
            )
        return f"## Error\n\n`{err}`"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process(image):
    """YOLO detection -> cost estimation -> LLM analysis."""
    if image is None:
        return None, "Please upload an image."

    pil_image = Image.fromarray(image)
    h, w = image.shape[:2]

    # --- Stage 1: YOLO detection ---
    detections, annotated_np = run_yolo(pil_image)
    yolo_available = detections is not None

    cost_result = None
    yolo_context = ""
    if yolo_available and detections:
        cost_result = estimate(detections, w, h)
        yolo_context = format_yolo_context(detections, cost_result)
    elif yolo_available and not detections:
        yolo_context = "YOLO Detection: No damage objects detected by the model."

    # --- Stage 2: LLM analysis ---
    llm_available = bool(API_KEY)

    if llm_available:
        llm_report = analyze_damage(pil_image, yolo_context=yolo_context)
    else:
        llm_report = ""

    # --- Compose final report ---
    report_parts = []

    if yolo_available and detections and cost_result:
        report_parts.append("## YOLO Detection Summary\n")
        report_parts.append(cost_result["summary"])
        report_parts.append("\n---\n")

    if llm_report:
        if yolo_available and detections:
            report_parts.append("## AI Vision Analysis (LLM-Refined)\n")
        report_parts.append(llm_report)
    elif not llm_available and not yolo_available:
        report_parts.append(
            "## No Analysis Available\n\n"
            "Neither the YOLO model nor the Groq API key is available.\n"
            "- Place model weights at the expected path\n"
            "- Or set `GROQ_API_KEY` environment variable"
        )
    elif not llm_available and yolo_available and not detections:
        report_parts.append(
            "## YOLO Result\n\n"
            "No damage detected by YOLO model. "
            "Set `GROQ_API_KEY` for deeper LLM-based analysis."
        )

    final_report = "\n".join(report_parts)
    display_image = annotated_np if annotated_np is not None else image

    return display_image, final_report


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Car Damage Cost Estimator") as demo:
    gr.Markdown(
        """
        # Car Damage Detection & Cost Estimator
        Upload a photo of a damaged car. The AI will detect all damage and estimate repair costs in **Indian Rupees (INR)**.
        > Powered by YOLOv8 Detection + Llama 4 Scout Vision (via Groq)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Car Image", type="numpy")
            run_btn = gr.Button("Analyze Damage", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Detected Damage (YOLO)", type="numpy", interactive=False
            )

    cost_output = gr.Markdown(label="Damage Assessment Report")

    run_btn.click(
        fn=process,
        inputs=[input_image],
        outputs=[output_image, cost_output],
    )

    sample_dir = os.path.join(os.path.dirname(__file__), "samples")
    test_images = glob.glob(os.path.join(sample_dir, "*.jpg"))[:6]
    if test_images:
        gr.Examples(
            examples=[[f] for f in test_images],
            inputs=[input_image],
            label="Sample Images from Test Set",
        )


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())
