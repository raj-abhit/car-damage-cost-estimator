"""
Rule-based cost estimator.
Takes YOLO detections → returns itemised cost breakdown + total range.
All costs in Indian Rupees (INR) based on 2024-2025 Indian market rates.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Base repair cost ranges (INR) per damage type
# Aligned with the 4 YOLO dataset classes:
#   broken-glass, deformation, rust, scratch
# ---------------------------------------------------------------------------
BASE_COST = {
    "broken-glass": (5_000, 15_000),   # windshield / window replacement
    "deformation":  (8_000, 40_000),   # dent, panel beating, bumper damage
    "rust":         (3_000, 12_000),   # rust treatment + repaint
    "scratch":      (1_500,  8_000),   # scratch repair + touch-up paint
}

# Severity multipliers based on damage area (% of image)
# area_fraction = bbox_area / image_area
def _severity(area_fraction: float) -> Tuple[str, float]:
    if area_fraction < 0.03:
        return "minor",    0.5
    elif area_fraction < 0.10:
        return "moderate", 1.0
    elif area_fraction < 0.25:
        return "severe",   1.5
    else:
        return "critical", 2.0


@dataclass
class DamageItem:
    label: str
    confidence: float
    area_fraction: float          # ratio of damage area to full image
    severity: str = field(init=False)
    cost_low: float = field(init=False)
    cost_high: float = field(init=False)

    def __post_init__(self):
        self.severity, mult = _severity(self.area_fraction)
        base_low, base_high = BASE_COST.get(self.label, (100, 500))
        self.cost_low  = round(base_low  * mult)
        self.cost_high = round(base_high * mult)


def estimate(detections: List[dict], image_w: int, image_h: int) -> dict:
    """
    detections: list of dicts with keys:
        - label (str)
        - confidence (float 0-1)
        - mask_area (float, pixel count of segmentation mask)  OR
        - box (x1, y1, x2, y2) if mask not available

    Returns a dict with:
        - items: list of DamageItem
        - total_low / total_high: summed cost range
        - summary: human-readable string
    """
    image_area = image_w * image_h
    items: List[DamageItem] = []

    for det in detections:
        label = det["label"]
        conf  = det["confidence"]

        # Area fraction — prefer mask area, fall back to bounding box
        if "mask_area" in det and det["mask_area"]:
            area_frac = det["mask_area"] / image_area
        elif "box" in det:
            x1, y1, x2, y2 = det["box"]
            area_frac = ((x2 - x1) * (y2 - y1)) / image_area
        else:
            area_frac = 0.05  # default moderate

        items.append(DamageItem(label=label, confidence=conf, area_fraction=area_frac))

    if not items:
        return {
            "items": [],
            "total_low": 0,
            "total_high": 0,
            "summary": "No damage detected. Vehicle appears to be in good condition.",
        }

    total_low  = sum(i.cost_low  for i in items)
    total_high = sum(i.cost_high for i in items)

    lines = ["**Damage Assessment Report**\n"]
    for i, item in enumerate(items, 1):
        lines.append(
            f"{i}. **{item.label.replace('-', ' ').title()}**  "
            f"({item.severity.upper()}, {item.confidence*100:.0f}% confidence)\n"
            f"   Estimated repair: **Rs {item.cost_low:,} – Rs {item.cost_high:,}**"
        )

    lines.append(f"\n---\n**Total Estimated Repair Cost: Rs {total_low:,} – Rs {total_high:,}**")
    lines.append(
        "\n*(Costs are approximate based on Indian market rates for local/aftermarket parts. "
        "Authorised service center costs may be 1.5x–2x higher.)*"
    )

    return {
        "items": items,
        "total_low": total_low,
        "total_high": total_high,
        "summary": "\n".join(lines),
    }
