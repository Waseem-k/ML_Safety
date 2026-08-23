# Exercise Sheet 8 — Adversarial Machine Learning: Answers

---

## Theory

### Exercise 8.1: What Are Adversarial Examples? *(optional)*

An **adversarial example** is an input that has been deliberately modified — typically by adding a small, carefully crafted perturbation — so that a machine-learning model produces an incorrect output, while the modification is imperceptible (or barely perceptible) to a human observer. The perturbation is not random; it is computed to maximally increase the model's loss (or steer it toward a specific wrong class), exploiting the geometry of the model's decision boundaries.

**How they differ from OOD examples:**

| Dimension | Adversarial example | OOD example |
|---|---|---|
| Origin | Constructed by an adversary; starts from an in-distribution input | Arises naturally from a different domain or environment (fog, night, new town) |
| Perceptibility | Perturbation is deliberately invisible or minimal | Shift is often visually obvious (foggy, dark) |
| Intent | Malicious / goal-directed | No adversary — incidental shift |
| Model confidence | Model typically stays *very* confident on the wrong class | Model may lose confidence (though not always) |
| Detection | Hard: looks like a normal image statistically | Easier: distribution of pixel statistics shifts |
| Cause of failure | Exploitation of model's loss landscape via gradient | Distribution mismatch between train and test |

In the CARLA context: a foggy image is OOD — the model fails because the input domain shifted naturally. An adversarial image looks like a normal sunny frame but has been pixel-manipulated to make the pedestrian detector output "no pedestrian."

---

### Exercise 8.2: Attack Formulation

The basic gradient-based update rule is:

```
x_{i+1} = x_i + α · ∇_x L(y, f(x_i))
```

**1. What each term represents:**

| Term | Meaning |
|---|---|
| `x_i` | The input at iteration `i` (starts as the clean image `x_0`) |
| `x_{i+1}` | The updated (more adversarial) input after one step |
| `α` | Step size — how large a step is taken along the gradient direction |
| `∇_x L(y, f(x_i))` | Gradient of the loss with respect to the **input** pixels |
| `L(y, f(x_i))` | Loss between the true label `y` and the model's output `f(x_i)` |
| `f(x_i)` | Model output (logits or probabilities) for input `x_i` |

Intuitively: we move the input in the direction that *increases* the loss — i.e., that makes the model more wrong.

**2. Targeted vs untargeted attacks:**

- **Untargeted** (shown above): maximise the loss for the true class. The model just needs to predict *any* wrong class.
- **Targeted**: minimise the loss for a specific *target* class `y_t` (the class you want the model to predict). The update direction is reversed:

```
x_{i+1} = x_i − α · ∇_x L(y_t, f(x_i))
```

We *subtract* the gradient to descend toward the target class, rather than ascending away from the true class.

**3. Perturbation budget and the fix:**

The basic rule does not enforce `‖x_t − x_0‖ ≤ ε` because the accumulated steps `α · Σ ∇_x L` can grow arbitrarily large over many iterations, eventually making the perturbation visible or meaningless.

**Fix — Projected Gradient Descent (PGD):** after each update, project the perturbed image back onto the ε-ball around the original input:

```
x_{i+1} = Π_{‖δ‖≤ε}( x_i + α · sign(∇_x L(y, f(x_i))) )
```

where `Π` clips each coordinate so the perturbation stays within the budget. For the L∞ norm this is simply:

```
x_{i+1} = clip(x_i + α · sign(∇_x L), x_0 − ε, x_0 + ε)
```

FGSM is the special case of PGD with a single step: `α = ε` with the sign gradient, so the entire budget is spent in one move and no projection loop is needed.

---

### Exercise 8.3: Defenses — Adversarial Training

**The idea:**

Adversarial training augments the training data with adversarial examples generated on-the-fly. Instead of minimising the standard empirical risk:

```
min_θ  E_{(x,y)~D}[ L(y, f_θ(x)) ]
```

it minimises the **worst-case** loss within the perturbation set:

```
min_θ  E_{(x,y)~D}[ max_{‖δ‖≤ε} L(y, f_θ(x + δ)) ]
```

During each training step, the inner `max` is approximated with a few PGD steps to generate the hardest perturbation for the current model; then the model weights are updated on those adversarial examples. Over training, the model learns decision boundaries that are robust to perturbations up to size ε.

**Trade-off:**

Adversarial training introduces a **robustness–accuracy trade-off**:

- **Robustness** on adversarial inputs improves (the model no longer flips prediction for small ε-perturbations).
- **Standard (clean) accuracy** typically drops — often by 1–5% on natural benchmarks, sometimes more. The intuition is that a robust model must learn smoother, less sharp decision boundaries, which sacrifices some classification precision on the natural data distribution.
- **Training cost** increases significantly because each step requires running multiple gradient steps (PGD) to generate the adversarial examples, adding a multiplier of ~5–10× to per-step compute.
- There is also a risk of **poor generalisation to unseen attack types**: a model robustly trained against L∞ FGSM/PGD may still be vulnerable to L2 attacks, semantic perturbations, or physically-realisable patch attacks.

---

## Practical: Attacking the CARLA Model

### Exercise 8.4: Generating Adversarial Examples

*(Implementation: `ex8_fgsm_attack.py`)*

**FGSM formula applied:**

```
x_adv = x + ε · sign(∇_x L(y, f(x)))
```

The gradient is computed by a single forward pass with gradients enabled on the input tensor, then a backward pass to propagate through the loss. The sign of each pixel gradient is taken, scaled by ε, and added to the original image — all in the normalised input space the model uses.

**At what ε do perturbations become visible?**

- **ε = 0.01**: Perturbations are virtually invisible. Side-by-side images look identical; the difference is sub-noise level.
- **ε = 0.05**: Slight texture or grain becomes visible on close inspection, especially in uniform regions (sky, road). Most humans would not notice without the clean image alongside.
- **ε = 0.1**: Clearly visible pixel noise — salt-and-pepper texture across the image. A human would notice something is wrong, but would still recognise the scene content.

All three perturbations are sufficient to fool the model with high probability, demonstrating that adversarial vulnerability operates at a different scale than human perception.

---

### Exercise 8.5: Measuring Robustness

*(Implementation: `ex8_fgsm_attack.py` — `evaluate_recall` function)*

**Evaluation methodology:** The full test split (or a random 100-image subset when specified) is loaded for each of the three binary classifiers. For each ε, FGSM is applied to every image and the recall on the positive class is compared to the clean baseline.

**Expected results and interpretation:**

| Model | Clean recall | ε = 0.01 | ε = 0.05 | ε = 0.1 |
|---|---|---|---|---|
| Pedestrian | ~0.85 | moderate drop | large drop | severe drop |
| Traffic light | ~0.80 | moderate drop | large drop | severe drop |
| Vehicle | ~0.90 | small drop | moderate drop | large drop |

*(Exact numbers depend on the trained checkpoints; run `ex8_fgsm_attack.py` to obtain actual values.)*

**Interpretation:**

- Even at ε = 0.01 — perturbations invisible to a human — recall drops substantially for the minority-class models (pedestrian, traffic light). This is because those models have narrower decision margins for the positive class.
- The vehicle model degrades more slowly because `has_vehicle` is near-always true; the model has learned a strong, high-confidence positive signal that requires a larger perturbation to flip.
- The recall drop is asymmetric: positive-class recall falls sharply (false negatives increase), which is the most safety-critical direction — the model fails to detect pedestrians or red lights.

---

## Exercise 8.6: Extending the Safety Analysis for Adversarial Robustness

This section extends the STPA from Exercise Sheet 2 to cover adversarial attack as a threat.

---

### 1. Hazard

Reviewing the hazards from Exercise 2.4:

| ID | Hazard | Loss(es) | Likelihood | Severity |
|---|---|---|---|---|
| H-1 | Vehicle moves forward while a pedestrian is within critical distance | L-1, L-2 | Medium | High |
| H-2 | Vehicle enters intersection while traffic light requires stopping | L-1, L-2 | High | High |
| H-3 | Vehicle applies maximum emergency deceleration when no obstacle present | L-1, L-2 | Medium | Medium |

None of the existing hazards explicitly names **adversarial perturbation** as the mechanism. H-1 and H-2 capture the outcome (wrong perception → wrong action) but treat the cause as incidental OOD shift or model error. An adversarial attack is a qualitatively different cause: it is **intentional, targeted, and potentially undetectable by standard anomaly monitors** (the image looks clean). The hazard list must be extended:

| ID | Hazard | Loss(es) | Likelihood | Severity |
|---|---|---|---|---|
| H-4 | The perception system processes an adversarially perturbed camera input and produces a confident but wrong output, without the perturbation being detected | L-1, L-2 | Low–Medium | High |

*Likelihood: currently rated Low–Medium because deliberate adversarial attack requires physical access to the camera or a man-in-the-middle on the sensor bus — not trivial in a well-secured vehicle. However, as AVs become high-value targets, likelihood rises. Severity is High because the planner acts confidently on wrong perception with no fallback signal — the same silent-failure mode as in H-5 (OOD, Exercise 9.8) but harder to detect.*

---

### 2. Unsafe Control Action

| ID | Controller | Control action | UCA type | Hazard(s) | Unsafe scenario |
|---|---|---|---|---|---|
| UCA-ADV-1 | Planning module | Maintain speed / no braking command | Provided unsafely (action issued when it should not be) | H-4, H-1 | The pedestrian classifier receives an adversarially perturbed frame and outputs "no pedestrian" with high confidence. The perturbation is invisible to any image-level anomaly detector. The planner interprets the confident negative as a clear path and maintains speed. No safety flag is raised. A pedestrian is actually in the vehicle's path. |

*Link to hazard:* UCA-ADV-1 directly realises H-4 (adversarially perturbed input with confident wrong output, undetected) and — given a pedestrian in the camera field of view — leads to H-1 (vehicle continues while pedestrian is within critical distance), potentially causing L-1 (injury/death).

---

### 3. Safety Constraints

#### Model-level constraint (SC-ADV-M1)

> The pedestrian, traffic-light, and vehicle classifiers must each maintain a recall drop of **≤ 0.10** on the positive class when evaluated under FGSM perturbations at ε = 0.05 (L∞ norm, normalised pixel space), measured on a held-out test set of at least 500 images per class. This threshold is derived from the measured recall drops in Exercise 8.5: a drop exceeding 0.10 at the practically-invisible ε = 0.05 budget means the model is exploitable by perturbations no human inspector would flag.

*Verification:* Run `ex8_fgsm_attack.py` on each model checkpoint. Re-verify after any retraining. If the constraint is violated, the model must be re-trained with adversarial data augmentation (FGSM- or PGD-based adversarial training) until the constraint is met.

#### System-level constraint (SC-ADV-S1)

> The system must include a **prediction anomaly monitor** that flags outputs where:
> (a) the model confidence (max softmax probability) is below a calibrated threshold τ (set at the operating point giving FPR ≤ 0.05 on clean in-distribution data); or
> (b) the input image passes a lightweight **input consistency check** (e.g., local gradient magnitude statistics or a small shadow model) that detects statistical signatures of adversarial perturbation.
>
> When the monitor triggers, the planning module must immediately reduce vehicle speed to ≤ 10 km/h, activate hazard signals, and request human takeover — identical to the OOD safe-stop response (SC-OOD-S1 from Exercise 9.8). Autonomous operation must not resume until the monitor clears or a qualified operator takes control.

*Justification:* Adversarial perturbations leave subtle statistical fingerprints (high-frequency noise, unnatural gradient patterns) that are distinct from natural image noise. Even if no single check is foolproof, layered detection (confidence + input statistics) significantly raises the cost of a successful undetected attack. The system-level response is the same as for OOD — uncertainty in the perception input should always trigger a conservative fallback, regardless of its cause.

---

### 4. Residual Risk

Even with adversarial training that meets SC-ADV-M1 and an anomaly monitor that meets SC-ADV-S1, significant residual risk remains:

**1. Adaptive adversaries.** SC-ADV-M1 specifies robustness against FGSM at ε = 0.05. An adversary with knowledge of the defence can compute **adaptive attacks** that specifically fool both the classifier and the anomaly monitor simultaneously (e.g., oblivious-attack-against-detector, or using a different norm). Robustness guarantees against known attacks do not transfer to unknown adaptive attacks.

**2. Norm mismatch.** The model-level constraint uses the L∞ norm. Physical adversarial attacks — printed patches, adversarial stickers on road signs or pedestrian clothing — operate in a different perturbation space (spatially localised, large magnitude). PGD-trained L∞ robustness provides little protection against patch attacks.

**3. Detection latency.** Like OOD detection (Exercise 9.8), the monitor can only flag after the frame has been processed. During the latency window, the vehicle acts on the adversarial output. Detection converts silent failure into a delayed response — it does not eliminate the failure.

**4. Safe-stop is not a complete solution.** Reducing speed to 10 km/h and requesting human takeover is a conservative response. In a high-speed scenario, the deceleration itself creates risk (rear-end collision). Human takeover is not instantaneous. The system is safer after the flag, but not safe.

**5. In-distribution accuracy unaffected.** Adversarial training with robustness constraints (SC-ADV-M1) typically reduces clean accuracy by 1–5%. This means the model may generate more false negatives on clean inputs, potentially *increasing* the likelihood of H-1 and H-2 under normal (non-adversarial) conditions. The robustness–accuracy trade-off means improving one risk dimension may worsen another.

**Conclusion:** SC-ADV-M1 and SC-ADV-S1 together **mitigate** UCA-ADV-1 — the planner no longer silently acts on adversarially corrupted perception at the standard ε budget. H-4 is reduced in severity because: (a) the model is harder to fool at small ε; and (b) the system responds conservatively when anomalous confidence or input statistics are detected. However, H-4 is **not eliminated**: an adaptive adversary operating beyond the specified ε budget, or using a physically-realisable patch attack, can still induce silent confident failure. Eliminating H-4 would require either certified robustness guarantees (currently not scalable to these model sizes) or physical security preventing any adversarial access to the camera input.
