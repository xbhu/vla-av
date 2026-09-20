"""
Cluster C2 (Reasoning-Action Coherence) shared utilities.

Design idea:
1. C1's debug/full-battery data showed that under the baseline condition
   the model almost always takes a "shortcut" (just emitting a generic
   "This is a straightforward scenario..." template with no concrete
   decision statement), so C2's original plan of "passively observing
   natural CoT" had no usable samples.
2. Fix: reuse the predict_c1(forced_cot_prefix=...) mechanism already
   built for C1. Here we prefill "This is a complex scenario requiring
   additional reasoning." to force the model down the full reasoning
   path -- this is still pure left-to-right autoregressive generation;
   the model does not know in advance what it will write or what the
   final trajectory will be. We are only choosing which door it walks
   through, not handing it the answer.
3. After generation:
   a) Parse the lateral/longitudinal action category the model itself
      states in the CoT text.
   b) Use a geometry/kinematics-based classifier, independent of the
      model, to map the decoded trajectory onto the same 12-class
      vocabulary.
   c) Compare the two and report agreement, separately for lateral and
      longitudinal.

=== 2026-09-19 fix notes ===
Running --debug on 5 scenes revealed that the model's actual section
heading is "### Best Driving Action" (or **Best Driving Action**), not
"Final Action Decision" as originally assumed. The old regex matched
nothing, so parsing always fell back to scanning the full CoT text --
which picked up phrases like "vehicles must stop" from the "Critical
Object Description" section (describing a red traffic light in the
environment), misreading them as the model's own stated decision. This
produced a large number of false-positive "contradictions".

Fixed behavior:
  - Match the "Best Driving Action" heading (case-insensitive, tolerant
    of markdown ### and **bold**, and tolerant of the decision being
    stated inline on the heading line itself, e.g. "**Best Driving
    Action: Move forward with a deceleration...**").
  - The search window for keyword matching is [end of heading, </think>)
    -- it never leaks forward into the raw action-token text after
    <answer>, and never leaks backward into unrelated sections like
    Scene Description.
  - If the heading truly does not appear (some scenes jump straight from
    "Reasoning on Intent" into action tokens with no explicit decision
    statement), this is now explicitly flagged as no_decision_stated=True
    instead of falling back to a full-text scan.
  - If longitudinal resolves to "stop" but no lateral direction word is
    found, that's expected (a stopped vehicle has no meaningful "which
    way am I going" answer) -- flagged as lateral_is_na_due_to_stop=True
    so downstream analysis treats it as "not applicable", not as a
    parser failure to be silently dropped.
"""

import re
import numpy as np

# ---------------------------------------------------------------------
# Prefill text used to force the model down the full reasoning path
# ---------------------------------------------------------------------
FORCED_FULL_COT_PREFIX = (
    "<think>\n"
    "This is a complex scenario requiring additional reasoning.\n"
)

# ---------------------------------------------------------------------
# 1. Parse the model's stated action category from the CoT text
# ---------------------------------------------------------------------

# More specific patterns listed first, so "turn left" doesn't grab text
# that actually says "change lane to the left"
LATERAL_KEYWORD_PATTERNS = [
    ("change lane to left", [r"change\s+lane\s+to\s+the?\s*left", r"changing\s+lane\s+to\s+the?\s*left", r"lane\s+change\s+to\s+the?\s*left"]),
    ("change lane to right", [r"change\s+lane\s+to\s+the?\s*right", r"changing\s+lane\s+to\s+the?\s*right", r"lane\s+change\s+to\s+the?\s*right"]),
    ("turn left", [r"turn(?:ing)?\s+left"]),
    ("turn right", [r"turn(?:ing)?\s+right"]),
    ("move forward", [r"move\s+forward", r"moving\s+forward", r"continue\s+forward", r"keep\s+forward", r"go(?:ing)?\s+straight", r"proceed(?:ing)?\s+forward"]),
]

LONGITUDINAL_KEYWORD_PATTERNS = [
    ("quick deceleration", [r"quick(?:ly)?\s+deceler", r"rapid(?:ly)?\s+deceler", r"sharp(?:ly)?\s+deceler"]),
    ("quick acceleration", [r"quick(?:ly)?\s+acceler", r"rapid(?:ly)?\s+acceler", r"sharp(?:ly)?\s+acceler"]),
    ("deceleration to zero", [r"deceler\w*\s+to\s+zero", r"come\s+to\s+a\s+stop", r"slow(?:ing)?\s+to\s+a\s+stop"]),
    ("stop", [r"\bstop\b", r"remain(?:ing)?\s+stopped", r"stay(?:ing)?\s+stopped", r"stationary"]),
    ("maintain constant speed", [r"constant\s+speed", r"maintain\w*\s+speed", r"steady\s+speed"]),
    ("deceleration", [r"deceler"]),
    ("acceleration", [r"acceler"]),
]

# Matches the "Best Driving Action" heading in whatever form the model
# writes it: markdown ### prefix, **bold** wrapping, optional colon,
# and it may or may not carry inline content after the colon.
BEST_ACTION_MARKER_RE = re.compile(
    r"#{1,4}\s*\*{0,2}\s*best\s+driving\s+action\s*:?\s*\*{0,2}",
    re.IGNORECASE,
)


def _match_first(text, patterns):
    for label, regex_list in patterns:
        for pat in regex_list:
            if re.search(pat, text, re.IGNORECASE):
                return label
    return None


def parse_stated_category(cot_text):
    """
    Parse the lateral/longitudinal action category the model states for
    itself in the generated CoT text.

    Returns: dict {
        "lateral": str or None,
        "longitudinal": str or None,
        "used_final_action_section": bool,  # True if the "Best Driving
            Action" heading was found and the search window was scoped
            to it. False means no such heading exists in this CoT at
            all (see no_decision_stated below) -- there is no
            full-text-scan fallback anymore.
        "no_decision_stated": bool,  # True if the model never wrote an
            explicit "Best Driving Action" section for this scene.
        "lateral_is_na_due_to_stop": bool,  # True if longitudinal
            resolved to "stop" and no lateral direction word was found
            -- this is an expected N/A, not a parse failure.
    }
    """
    if not cot_text:
        return {
            "lateral": None,
            "longitudinal": None,
            "used_final_action_section": False,
            "no_decision_stated": True,
            "lateral_is_na_due_to_stop": False,
        }

    marker_match = BEST_ACTION_MARKER_RE.search(cot_text)
    if not marker_match:
        return {
            "lateral": None,
            "longitudinal": None,
            "used_final_action_section": False,
            "no_decision_stated": True,
            "lateral_is_na_due_to_stop": False,
        }

    end_idx = cot_text.find("</think>")
    if end_idx == -1:
        end_idx = len(cot_text)
    window = cot_text[marker_match.end():end_idx]

    lateral = _match_first(window, LATERAL_KEYWORD_PATTERNS)
    longitudinal = _match_first(window, LONGITUDINAL_KEYWORD_PATTERNS)

    lateral_is_na_due_to_stop = (lateral is None and longitudinal == "stop")

    return {
        "lateral": lateral,
        "longitudinal": longitudinal,
        "used_final_action_section": True,
        "no_decision_stated": False,
        "lateral_is_na_due_to_stop": lateral_is_na_due_to_stop,
    }


# ---------------------------------------------------------------------
# 2. Model-independent geometry/kinematics trajectory -> category
#    classifier.
#    !! The thresholds below are first-pass placeholder values, not yet
#    calibrated against real data. After running debug/full battery,
#    revisit these against the actual heading_change / lateral_offset /
#    delta_v distributions to check they're reasonable !!
# ---------------------------------------------------------------------

HEADING_TURN_THRESHOLD_DEG = 20.0     # heading change beyond this counts as a "turn", needs calibration
LATERAL_OFFSET_LANECHANGE_M = 1.5     # small heading change but lateral offset beyond this counts as "lane change", needs calibration

STOP_SPEED_THRESHOLD_MPS = 0.5        # end speed below this counts as near-stopped, needs calibration
QUICK_DELTA_V_THRESHOLD_MPS = 3.0     # speed change beyond this counts as "quick", needs calibration
MODERATE_DELTA_V_THRESHOLD_MPS = 1.0  # speed change beyond this (but under quick) counts as ordinary accel/decel, needs calibration


def _trajectory_xy(trajectory):
    import torch
    if torch.is_tensor(trajectory):
        trajectory = trajectory.cpu().numpy()
    return np.asarray(trajectory)[:, :2]


def classify_trajectory(trajectory, interval_length=0.5):
    """
    trajectory: [num_poses, >=2] trajectory points in the ego local frame
                (convention: x = forward, y = left-positive; if the real
                data uses the opposite sign convention, verify against
                debug output and flip accordingly).
    Returns: dict {"lateral": str, "longitudinal": str, "heading_change_deg": float,
                "lateral_offset_m": float, "delta_v_mps": float}
    """
    xy = _trajectory_xy(trajectory)
    n = len(xy)
    if n < 2:
        return {"lateral": None, "longitudinal": None,
                "heading_change_deg": None, "lateral_offset_m": None, "delta_v_mps": None}

    # --- Lateral: overall drift angle from start to end + endpoint lateral offset ---
    start, end = xy[0], xy[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    drift_angle_deg = np.degrees(np.arctan2(dy, dx))
    lateral_offset_m = float(dy)

    if abs(drift_angle_deg) >= HEADING_TURN_THRESHOLD_DEG:
        lateral = "turn left" if drift_angle_deg > 0 else "turn right"
    elif abs(lateral_offset_m) >= LATERAL_OFFSET_LANECHANGE_M:
        lateral = "change lane to left" if lateral_offset_m > 0 else "change lane to right"
    else:
        lateral = "move forward"

    # --- Longitudinal: compare speed of the first segment vs the last segment ---
    seg_dists = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    seg_speeds = seg_dists / interval_length  # m/s
    v_start = float(seg_speeds[0])
    v_end = float(seg_speeds[-1])
    delta_v = v_end - v_start

    if v_end < STOP_SPEED_THRESHOLD_MPS:
        if v_start < STOP_SPEED_THRESHOLD_MPS:
            longitudinal = "stop"
        else:
            longitudinal = "deceleration to zero"
    elif delta_v <= -QUICK_DELTA_V_THRESHOLD_MPS:
        longitudinal = "quick deceleration"
    elif delta_v >= QUICK_DELTA_V_THRESHOLD_MPS:
        longitudinal = "quick acceleration"
    elif delta_v <= -MODERATE_DELTA_V_THRESHOLD_MPS:
        longitudinal = "deceleration"
    elif delta_v >= MODERATE_DELTA_V_THRESHOLD_MPS:
        longitudinal = "acceleration"
    else:
        longitudinal = "maintain constant speed"

    return {
        "lateral": lateral,
        "longitudinal": longitudinal,
        "heading_change_deg": drift_angle_deg,
        "lateral_offset_m": lateral_offset_m,
        "delta_v_mps": delta_v,
    }
