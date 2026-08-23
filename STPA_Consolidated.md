# Consolidated STPA Analysis — CARLA Autonomous Driving Perception System

Merges the base STPA from Exercise 2 with the adversarial-robustness extension (Exercise 8.6) and the OOD extension (Exercise 9.8) into single master tables.

---

## 1. Losses

| ID | Loss Description | Why Unacceptable |
|---|---|---|
| L-1 | Injury or death to pedestrians, the human safety operator, or other road users. | Human life is irreplaceable; preserving it is the primary safety mandate. |
| L-2 | Material damage to the AV prototype, other vehicles, or infrastructure. | Massive financial cost and destruction of the hardware prototype. |
| L-3 | Traffic violation (e.g., running a red light) without collision. | Legal/regulatory exposure; erodes public trust in AV safety even absent physical harm. *(Introduced implicitly in Ex. 9.8's H-5 losses.)* |

---

## 2. Hazards

| ID | Hazard | Loss(es) | Likelihood | Severity | Source |
|---|---|---|---|---|---|
| H-1 | The vehicle moves forward while a pedestrian is within the critical distance. | L-1, L-2 | Medium | High | Ex. 2.4 |
| H-2 | A vehicle enters an intersection while a traffic light requires stopping. | L-1, L-2 | High | High | Ex. 2.4 |
| H-3 | The vehicle abruptly applies maximum emergency deceleration when no obstacle is present. | L-1, L-2 | Medium | Medium | Ex. 2.4 |
| H-4 | The perception system processes an adversarially perturbed camera input and produces a confident but wrong output, without the perturbation being detected. | L-1, L-2 | Low–Medium | High | Ex. 8.6 |
| H-5 | The perception system operates on a camera input that is outside the ODD and this is not detected, causing the planner to act on unreliable perception output. | L-1, L-2, L-3 | High | High | Ex. 9.8 |

**Note on H-4 vs. H-5:** both describe a *silent failure* mode (confident-but-wrong perception with no fallback signal), but the causal mechanism differs — H-5 is incidental distribution shift (fog, night, novel town), while H-4 is an intentional, targeted perturbation. This distinction matters for detection strategy: H-5 can be caught by confidence/feature-based statistical drift; H-4 requires input-integrity checks because the image is statistically "clean."

---

## 3. Unsafe Control Actions (UCAs)

| ID | Controller | Control Action | UCA Type | Hazard(s) | Unsafe Scenario | Source |
|---|---|---|---|---|---|---|
| UCA-1 | Planning Module | Emergency Brake | Not provided | H-1 | The planner fails to issue an emergency brake command when a pedestrian is physically within the critical distance. | Ex. 2.6 |
| UCA-2 | Planning Module | Continue Driving | Provided unsafely | H-2 | The planner commands the vehicle to continue through a mapped intersection because the traffic-light detector failed to predict "present." | Ex. 2.6 |
| UCA-3 | Human Operator | Brake Override | Wrong timing | H-1, H-2 | The operator presses the brake pedal too late to avoid a collision after the planner fails to decelerate. | Ex. 2.6 |
| UCA-4 | Planning Module | Emergency Brake | Provided unsafely | H-3 | The planner issues an emergency brake, applying maximum deceleration due to a false positive from the vehicle detector. | Ex. 2.6 |
| UCA-OOD-1 | Planning Module | Maintain speed / no braking command | Provided unsafely | H-5, H-1 | The planner continues at speed while the camera input is out-of-ODD (fog, night, novel location) and the perception output is untrustworthy; the pedestrian detector outputs "no pedestrian" with high confidence and no OOD flag is raised. | Ex. 9.8 |
| UCA-ADV-1 | Planning Module | Maintain speed / no braking command | Provided unsafely | H-4, H-1 | The pedestrian classifier receives an adversarially perturbed frame and outputs "no pedestrian" with high confidence; the perturbation is invisible to image-level anomaly detectors, so no safety flag is raised while a pedestrian is actually in the vehicle's path. | Ex. 8.6 |

Both new UCAs share the same controller, control action, and UCA type as the original UCA-1 pattern — they are variants of "silent confident failure causing a missing brake command," differing only in the perceptual root cause (distribution shift vs. adversarial perturbation).

---

## 4. Safety Constraints

| UCA | Safety Constraint | Level | Verification | Source |
|---|---|---|---|---|
| UCA-1 | The perception model must accurately flag "pedestrian present" whenever a pedestrian enters the camera's field of view within the critical distance. | Model-level | Evaluate the pedestrian classifier's recall on a test set (Sheet 4). | Ex. 2.7 |
| UCA-2 | The system must have a reliable way to determine the *state* of a traffic light, not just its presence, before entering an intersection. | System-level | Requires architectural change (multi-class red/yellow/green model). | Ex. 2.7 |
| UCA-3 | The system must actively alert the operator to perception changes to mitigate delayed intervention during 4-hour shifts. | System-level | Addition of auditory/haptic alerts to the visual-only dashboard. | Ex. 2.7 |
| UCA-OOD-1 (model) — **SC-OOD-M1** | The OOD monitor must achieve recall ≥ 0.95 on fog/night/other clearly out-of-ODD inputs at FPR ≤ 0.05 on in-distribution sunny/daytime inputs. | Model-level | `ex9_msp_baseline.py` / `ex9_feature_ood.py` AUROC + operating-point evaluation on held-out OOD set. | Ex. 9.8 |
| UCA-OOD-1 (system) — **SC-OOD-S1** | On an OOD flag, the planner must reduce speed to ≤ 10 km/h within 3 s, activate hazard signals, and request human takeover; autonomous operation must not resume until cleared. | System-level | Integration test: inject synthetic OOD flag, verify planner state transition and timing. | Ex. 9.8 |
| UCA-ADV-1 (model) — **SC-ADV-M1** | Pedestrian/traffic-light/vehicle classifiers must maintain recall drop ≤ 0.10 on the positive class under FGSM at ε = 0.05 (L∞), measured on ≥ 500 held-out images per class. | Model-level | `ex8_fgsm_attack.py` recall-drop measurement; re-verify after retraining. | Ex. 8.6 |
| UCA-ADV-1 (system) — **SC-ADV-S1** | System must include a prediction anomaly monitor (confidence threshold + input-consistency check); on trigger, same safe-stop response as SC-OOD-S1. | System-level | Layered detection test (confidence threshold calibration + shadow-model consistency check) against known and held-out attacks. | Ex. 8.6 |

**Cross-cutting observation:** SC-OOD-S1 and SC-ADV-S1 specify an *identical* system-level fallback (safe-stop + human takeover). This is intentional — both hazards reduce to the same actionable signal ("perception cannot be trusted right now"), so the planner-side response only needs to be built once and triggered by either monitor. The two monitors differ, but the downstream mitigation is shared infrastructure.

---

## 5. Causal Loss Scenarios

| UCA | Causal Scenario | Root Cause | Related Constraint | Source |
|---|---|---|---|---|
| UCA-1 (not provided) | The pedestrian model outputs a false negative because sudden cloud cover alters lighting; the planner receives "no pedestrian" and continues driving. | Incorrect feedback due to OOD shift — model trained exclusively on sunny data. | Model-level (Accurate pedestrian detection). | Ex. 2.8 |
| UCA-2 | The traffic-light model correctly detects a green light and predicts "present"; the planner treats any "present" signal as a stop requirement (or vice versa). | Flawed internal planner logic — complex stop/go decision from insufficient binary (presence-only) feedback. | System-level (Determine traffic-light state). | Ex. 2.8 |
| UCA-3 | The automated system fails to brake; the operator does not override because they are looking away, suffering vigilance decrement late in a 4-hour shift. | Human factor (delayed intervention) exacerbated by lack of auditory alerts. | System-level (Active operator alerts). | Ex. 2.8 |
| UCA-OOD-1 | Fog degrades image contrast; MSP confidence stays misleadingly high (measured AUROC 0.49 on Fog — worse than random) because softmax overconfidence persists under blur-like shift; no OOD flag raised; planner continues at speed. | Detector choice — MSP is fundamentally the wrong tool for this shift type; feature-space (Mahalanobis) detection recovers AUROC 0.98 on the same scenario. | Model-level (SC-OOD-M1) — motivates dropping MSP for Mahalanobis in production. | Ex. 9.7 / ex9 results |
| UCA-ADV-1 | An adversary perturbs the pedestrian frame within ε = 0.05 (L∞); recall on the positive class collapses (measured: 0.4703 → 0.0014, a 0.4688 drop) while the perturbation stays sub-visible. | Gradient-based exploitation of the classifier's decision boundary; no adversarial training or input-consistency check in place. | Model-level (SC-ADV-M1) — current model is far outside the ≤0.10 drop budget. | Ex. 8.5 / ex8 results |

---

## 6. Empirical Grounding (from experiment logs)

These measured results justify the likelihood/severity ratings and constraint thresholds above:

**Adversarial (`ex8_run_log.txt`, `ex8_vehicle_remainder_log.txt`):**

| Model | Clean recall | ε=0.01 | ε=0.05 | ε=0.10 |
|---|---|---|---|---|
| has_pedestrian | 0.4703 | drop +0.4419 | drop +0.4688 | — |
| has_vehicle | 0.9044 | — | drop +0.8096 | drop +0.7215 |

Both classifiers collapse far past the SC-ADV-M1 budget (≤0.10 drop) at ε as small as 0.01–0.05 — confirming H-4/UCA-ADV-1 as a live, unmitigated hazard for the current checkpoints.

**OOD (`ex9_msp_run_log.txt`, `ex9_feature_run_log.txt`):**

| Scenario | MSP AUROC | Mahalanobis AUROC | Gap |
|---|---|---|---|
| Fog | 0.4932 | 0.9822 | +0.4891 |
| Night | 0.6367 | 0.9995 | +0.3628 |
| Town01 | 0.5512 | 0.7086 | +0.1574 |

MSP fails to meet SC-OOD-M1 (recall ≥0.95 @ FPR ≤0.05) on every scenario — it is at or below random-chance separation on Fog. Mahalanobis clears the bar on Fog and Night but still falls short on Town01, which — per the Ex. 9.5 ODD revision — is *in-distribution* and should be treated as a model capability gap, not an OOD-monitor failure.

---

## 7. Consolidated Residual Risk

1. **Detection latency** (both H-4 and H-5): a flag can only be raised after a frame is processed; the vehicle acts on the last output during that window.
2. **Silent-failure conversion, not elimination**: monitors turn silent failure into *delayed* failure — the underlying perception unreliability is not resolved, only flagged.
3. **Safe-stop carries its own risk**: emergency deceleration at speed risks rear-end collision; human takeover is not instantaneous.
4. **Adaptive/physical adversaries** (H-4-specific): SC-ADV-M1's L∞ FGSM robustness does not transfer to adaptive attacks or physically-realized patches.
5. **ODD boundary imprecision** (H-5-specific): Town01 is defined in-ODD but the model measurably underperforms there (Mahalanobis AUROC only 0.71) — a capability gap the OOD monitor is not designed to catch.
6. **Robustness–accuracy trade-off**: adversarial training to satisfy SC-ADV-M1 typically cuts clean accuracy 1–5%, which can raise the baseline likelihood of H-1/H-2 under normal conditions — improving one hazard can worsen another.

**Overall conclusion:** the five hazards are not independent — H-4 and H-5 are both instances of "planner acts on unreliable perception without a fallback signal," differing only in root cause (natural shift vs. adversarial). The system-level mitigation (SC-OOD-S1 / SC-ADV-S1) is effectively one shared safe-stop mechanism triggered by two different upstream monitors. None of H-1 through H-5 are eliminated by the proposed constraints — all are mitigated to "detected, conservative fallback" rather than "prevented."
