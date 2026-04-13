"""Rule-based estimator calibrated to Indian part + labour prices."""

from dataclasses import dataclass, field
from typing import List, Tuple


# Base parts + labour ranges in INR.
# Calibrated using 2025-2026 India references:
# - ZigWheels spare-parts tables (Swift/Dzire/i20/Creta/Fortuner)
# - AIS windshield MRP list (effective 1 Apr 2025)
# - GoMechanic bodywork and rust-treatment price cards (2026)
REPAIR_PROFILES = {
    # label:part_key -> (parts_low, parts_high, labour_low, labour_high)
    "scratch:spot": (600, 1500, 700, 1800),
    "scratch:panel": (900, 2200, 1800, 3500),
    "scratch:bumper": (1200, 3200, 2000, 4200),
    "deformation:bumper": (1800, 19000, 2200, 9000),
    "deformation:door_fender": (1200, 21000, 2200, 9500),
    "deformation:major_panel": (4000, 45000, 4500, 15000),
    "broken-glass:front_ws": (3500, 22000, 1200, 5000),
    "broken-glass:rear_ws": (2600, 22000, 1200, 5000),
    "broken-glass:door_glass": (700, 4500, 600, 2500),
    "rust:surface": (400, 1800, 1500, 4500),
    "rust:panel": (900, 3500, 2200, 7000),
    "rust:severe": (2500, 12000, 5000, 18000),
    "default:default": (1000, 5000, 1200, 4500),
}

PROFILE_HIGH_CAP = {
    "scratch:spot": 6500,
    "scratch:panel": 12000,
    "scratch:bumper": 15000,
    "deformation:bumper": 45000,
    "deformation:door_fender": 52000,
    "deformation:major_panel": 95000,
    "broken-glass:front_ws": 42000,
    "broken-glass:rear_ws": 42000,
    "broken-glass:door_glass": 11000,
    "rust:surface": 9000,
    "rust:panel": 18000,
    "rust:severe": 42000,
    "default:default": 25000,
}

SEVERITY_BAND = {
    "minor": 0.18,
    "moderate": 0.24,
    "severe": 0.30,
    "critical": 0.38,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _severity(area_fraction: float) -> Tuple[str, float]:
    """Severity and multiplier using bbox/mask area ratio."""
    if area_fraction < 0.01:
        return "minor", 0.75
    if area_fraction < 0.04:
        return "moderate", 1.00
    if area_fraction < 0.14:
        return "severe", 1.20
    return "critical", 1.45


def _infer_zone(box, image_w: int, image_h: int) -> str:
    """Infer coarse part location from bbox center."""
    if not box:
        return "center-mid"
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / max(image_w, 1)
    cy = ((y1 + y2) / 2) / max(image_h, 1)

    lr = "left" if cx < 0.33 else ("right" if cx > 0.67 else "center")
    fr = "front" if cy < 0.35 else ("rear" if cy > 0.68 else "mid")
    return f"{fr}-{lr}"


def _pick_profile(label: str, area_fraction: float, zone: str) -> str:
    fr = zone.split("-", 1)[0]

    if label == "broken-glass":
        if fr == "front":
            return "broken-glass:front_ws"
        if fr == "rear":
            return "broken-glass:rear_ws"
        return "broken-glass:door_glass"

    if label == "scratch":
        if area_fraction < 0.008:
            return "scratch:spot"
        if fr in {"front", "rear"}:
            return "scratch:bumper"
        return "scratch:panel"

    if label == "deformation":
        if area_fraction > 0.16:
            return "deformation:major_panel"
        if fr in {"front", "rear"}:
            return "deformation:bumper"
        return "deformation:door_fender"

    if label == "rust":
        if area_fraction < 0.02:
            return "rust:surface"
        if area_fraction < 0.10:
            return "rust:panel"
        return "rust:severe"

    return "default:default"


@dataclass
class DamageItem:
    label: str
    confidence: float
    area_fraction: float
    zone: str
    severity: str = field(init=False)
    profile: str = field(init=False)
    cost_low: int = field(init=False)
    cost_high: int = field(init=False)

    def __post_init__(self):
        self.severity, sev_mult = _severity(self.area_fraction)
        self.profile = _pick_profile(self.label, self.area_fraction, self.zone)

        parts_low, parts_high, labour_low, labour_high = REPAIR_PROFILES.get(
            self.profile, REPAIR_PROFILES["default:default"]
        )

        conf_mult = 0.85 + (0.25 * _clamp(self.confidence, 0.0, 1.0))
        raw_low = (parts_low + labour_low) * sev_mult * conf_mult
        raw_high = (parts_high + labour_high) * sev_mult * conf_mult

        # Keep a practical quote spread around the central estimate.
        mid = (raw_low + raw_high) / 2.0
        band = SEVERITY_BAND.get(self.severity, 0.28)
        # Lower confidence slightly widens the range, but keep bounded.
        band += (1.0 - _clamp(self.confidence, 0.0, 1.0)) * 0.08
        band = _clamp(band, 0.15, 0.45)

        low = mid * (1.0 - band)
        high = mid * (1.0 + band)

        self.cost_low = int(round(low))
        capped_high = min(high, PROFILE_HIGH_CAP.get(self.profile, PROFILE_HIGH_CAP["default:default"]))
        self.cost_high = int(round(max(capped_high, low + 400)))


def estimate(detections: List[dict], image_w: int, image_h: int) -> dict:
    image_area = max(image_w * image_h, 1)
    items: List[DamageItem] = []
    duplicate_counter = {}

    for det in detections:
        label = det["label"]
        conf = float(det["confidence"])

        if "mask_area" in det and det["mask_area"]:
            area_frac = float(det["mask_area"]) / image_area
        elif "box" in det:
            x1, y1, x2, y2 = det["box"]
            box_area = max((x2 - x1), 1) * max((y2 - y1), 1)
            area_frac = box_area / image_area
        else:
            area_frac = 0.03

        zone = _infer_zone(det.get("box"), image_w, image_h)
        item = DamageItem(
            label=label,
            confidence=conf,
            area_fraction=area_frac,
            zone=zone,
        )

        # Reduce over-counting when the same damage class repeats in nearby zones.
        key = (item.label, item.profile)
        seen = duplicate_counter.get(key, 0)
        if seen >= 1:
            damp = 0.85 if seen == 1 else 0.70
            item.cost_low = int(round(item.cost_low * damp))
            item.cost_high = int(round(item.cost_high * damp))
        duplicate_counter[key] = seen + 1

        items.append(item)

    if not items:
        return {
            "items": [],
            "total_low": 0,
            "total_high": 0,
            "summary": "No damage detected. Vehicle appears to be in good condition.",
        }

    total_low = sum(i.cost_low for i in items)
    total_high = sum(i.cost_high for i in items)

    lines = ["**Damage Assessment Report**\n"]
    for idx, item in enumerate(items, 1):
        lines.append(
            f"{idx}. **{item.label.replace('-', ' ').title()}** "
            f"({item.severity.upper()}, {item.confidence*100:.0f}% confidence, zone: {item.zone})\n"
            f"   Estimated repair: **Rs {item.cost_low:,} - Rs {item.cost_high:,}**"
        )

    lines.append(f"\n---\n**Total Estimated Repair Cost: Rs {total_low:,} - Rs {total_high:,}**")
    lines.append(
        "\n*(Costs are approximate for India and combine part + labour estimates. "
        "Authorised OEM service centers can be ~1.4x-2.2x higher than local repair quotes.)*"
    )

    return {
        "items": items,
        "total_low": total_low,
        "total_high": total_high,
        "summary": "\n".join(lines),
    }
