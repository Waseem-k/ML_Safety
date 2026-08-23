# Exercise Sheet 9 — Anomaly Detection: Answers

---

## Out-of-Distribution Detection (Theory)

### Exercise 9.1: The OOD Problem

**1. Why can a standard classifier not be trusted to signal when it receives an OOD input?**

A standard classifier is trained to partition its input space into known classes and to output a probability distribution over those classes via softmax. Because softmax is a normalization operation, it always produces values that sum to 1 — regardless of how far the input is from anything the model has ever seen. The model has no explicit representation of "unknown" or "outside training distribution." In high-dimensional space, the learned decision boundaries extend infinitely into regions that contain no training data, and the model can assign arbitrarily high confidence there. This phenomenon — high-confidence predictions on inputs that bear no resemblance to training data — has been demonstrated empirically: neural networks routinely assign >99% softmax confidence to random noise or adversarial out-of-distribution inputs. There is simply no mechanism built into the architecture or loss function that penalizes confident predictions on unseen input regions.

**2. Why is silent failure (confidently wrong prediction) worse than uncertain failure in a safety-critical system?**

In a safety-critical system like autonomous driving, the response to failure depends on the system knowing that something has gone wrong. Uncertain failure (low-confidence output) provides a detectable signal: monitors, thresholds, or the planning module itself can recognise that the perception output is unreliable and trigger a safe fallback — slow down, request human takeover, or stop. Silent failure provides no such signal. The planner receives a high-confidence, wrong prediction and acts on it as if it were correct, continuing at speed or failing to brake. No alarm is raised, no fallback is triggered, and the human operator has no indication that intervention is needed. The hazard is therefore not just the wrong perception output, but the fact that every downstream safety layer is bypassed by the apparent confidence. Silent failure is also much harder to detect in testing and fleet monitoring because the system looks fully functional right up to the accident.

---

### Exercise 9.2: Baseline OOD Detection — Maximum Softmax Probability (MSP)

**How it works:**

MSP (Hendrycks & Gimpel, 2017) uses the maximum value of the softmax output vector as an OOD score. The intuition is that in-distribution inputs produce peaked probability distributions (one class dominates), while OOD inputs produce flatter, more uniform distributions. Formally, for an input x with logits f(x), the OOD score is:

```
MSP(x) = max_c softmax(f(x))_c
```

A low MSP score (below a chosen threshold τ) flags the input as OOD; a high MSP score indicates in-distribution. For a binary classifier with a single sigmoid output p = σ(f(x)), MSP = max(p, 1−p).

**Main limitations:**

1. **Overconfidence on OOD inputs.** Neural networks frequently assign high softmax probabilities to inputs far outside the training distribution. The softmax magnifies logit differences exponentially, so even small systematic activation patterns on OOD inputs can produce high-confidence outputs. MSP therefore has poor separation between in-distribution and OOD scores in practice.

2. **Feature space is ignored.** MSP looks only at the final output. Two inputs can produce the same softmax output via completely different internal feature representations. An OOD input that happens to activate a class-like pattern in the output layer is indistinguishable from a genuine in-distribution input.

3. **Threshold calibration is difficult.** The threshold τ must be tuned on a held-out OOD set, but by definition we do not know in advance what OOD inputs look like in deployment. A threshold that works well for fog may not work for night or a different geographic location.

4. **No separation between aleatoric and epistemic uncertainty.** Low confidence can indicate either genuine ambiguity between classes (aleatoric) or an input that is simply novel (epistemic). MSP cannot distinguish the two.

---

### Exercise 9.3: Alternative Method — Mahalanobis Distance

**Method (Lee et al., 2018):**

Rather than looking at the output layer, the Mahalanobis distance method works in the feature space of the penultimate layer. The approach is:

1. **Fit a class-conditional Gaussian** on in-distribution training features. For each class c, compute the class mean μ_c from the extracted deep features. Estimate a single shared covariance matrix Σ (tied covariance) by pooling the centred features from all classes.

2. **Score test inputs** by computing the minimum Mahalanobis distance to any class centroid:

```
M(x) = min_c  (f(x) − μ_c)^T  Σ^{-1}  (f(x) − μ_c)
```

A large M(x) means the feature representation of x is far from all in-distribution class clusters — flagged as OOD.

**How it improves over MSP:**

- **Works in the feature space.** The penultimate layer captures learned, semantically meaningful representations. An OOD input whose final logits happen to resemble a known class will still produce a feature vector that lies far from the in-distribution clusters, so Mahalanobis can detect it where MSP cannot.
- **Accounts for feature correlations.** The inverse covariance Σ^{-1} normalises for the natural spread and correlation structure of in-distribution features, making the distance metric meaningful in high-dimensional space.
- **Not fooled by overconfident softmax.** Because the score is computed before the final classification layer, it is decoupled from the overconfidence problem that MSP inherits directly.
- **Empirically stronger.** Lee et al. showed that Mahalanobis distance substantially outperforms MSP in AUROC on standard benchmarks, and the advantage is largest on OOD scenarios that are semantically distant from the training distribution (e.g., different weather, different domain).

---

## Practical: OOD Detection for the CARLA Model

### Exercise 9.4: Visualising the Distribution Shift

*(Implementation: `ex9_visualise_shift.py`)*

**Part 2 — How the different-town images differ from training images, and how this compares to fog/night:**

The different-town (Town01) images are captured under identical lighting and weather to the training data — sunny, daytime, clear sky — but in a location with different road geometry, building architecture, vegetation, and lane markings. The visual difference is therefore **structural**: different textures, colours of buildings, road widths, and spatial arrangements of objects. The images are still sharp and well-lit; the pixel intensity distribution is similar to training images.

By contrast, fog and night images differ at the **signal level**: fog scatters light and dramatically reduces contrast and visibility across the entire image, while night images have low global illumination with localised light sources (street lamps, headlights). These conditions change the fundamental statistics of every pixel.

The fog/night shift is therefore a more severe, image-wide signal degradation, while the town shift is a subtler change in scene content. However, because our models rely on spurious spatial shortcuts (e.g., sky texture correlated with vehicle presence), even the structural town shift can degrade performance if those shortcuts do not transfer.

**Part 3 — Expected confidence behaviour:**

The models are expected to be less confident (lower mean MSP) on fog and night images, since the low-level features the models rely on are degraded or absent. Night images in particular should produce the lowest confidence. The different-town images, being closer in low-level statistics to training data, may preserve model confidence despite the structural shift — this would be a sign that confidence alone is insufficient to flag the town distribution shift.

---

### Exercise 9.5: Is the Different Town Out-of-Distribution?

**Part 1 — Does the original ODD decide the question?**

The ODD specified in Exercise 2.2 covered dimensions: weather (sunny/clear), lighting (daytime), camera condition (clean front-facing RGB), scene type (urban road, CARLA simulator), and vehicle speed. The scene type dimension likely said "urban road environment in CARLA simulator" without specifying which CARLA town. As written, the ODD is **ambiguous** about the different town: it permits the CARLA simulator and urban roads generally, but the model was only trained on one town (Town02). The ODD does not clearly decide whether a different town is inside or outside it.

**Part 2 — Revised ODD and choice:**

I choose to define the different town as **inside the ODD**, by revising the scene type dimension as follows:

> *Scene type — Operating condition:* Any CARLA urban road environment under nominal weather and lighting, regardless of specific town layout, road geometry, or building architecture. The system is required to operate correctly across all CARLA towns under nominal conditions.
> *Non-operating condition:* Non-urban environments, non-CARLA simulation environments, or any scene outside the CARLA simulator.

**Reason for this choice:** The system is intended to handle nominal driving conditions. Restricting deployment to a single geographic location (one CARLA town) would be an unreasonably narrow ODD for an autonomous driving system and would make the system useless in practice. The system should be robust to variation in scene content under nominal weather and lighting — that is the whole point of generalisation. Furthermore, the fog and night conditions represent genuine sensor degradation that warrants OOD flagging; a different road layout under clear conditions does not.

**Part 3 — Implication for treatment of town images:**

Since the different town is inside the ODD, the models are **required to handle those inputs correctly**. An OOD monitor must not flag them. If the models degrade on the different town, this is a **model capability gap** (insufficient training data coverage, overfitting to Town02 features), not an ODD violation. The appropriate response is to improve the models (more diverse training data, domain augmentation), not to expand the OOD monitor's rejection region to include inputs the system is supposed to handle.

---

### Exercise 9.6: Evaluating the MSP Baseline

*(Implementation: `ex9_msp_baseline.py`)*

The pedestrian model is used as the reference model throughout Exercises 9.6 and 9.7 for consistency.

**Expected results and interpretation:**

The MSP score distributions for fog and night are expected to show a leftward shift relative to in-distribution — more mass at lower confidence values — because the degraded image quality disrupts the features the model relies on. The separation should be cleaner for night (more extreme signal change) than for fog.

The AUROC measures how well MSP separates in-distribution from OOD images. An AUROC of 0.5 means the detector is no better than random; 1.0 means perfect separation. MSP typically achieves moderate AUROC on clear OOD scenarios (fog, night) but may struggle on the different-town scenario, where image statistics are close to training data and the model remains confident even on novel scenes.

---

### Exercise 9.7: Feature-Based OOD Detection

*(Implementation: `ex9_feature_ood.py`)*

**Method:** Mahalanobis distance on 512-dimensional features from the `avgpool` layer of the pedestrian ResNet-18. A class-conditional Gaussian with shared covariance is fitted on in-distribution training features. The OOD score for a test image is its minimum Mahalanobis distance to either class centroid.

**Expected comparison with MSP:**

The Mahalanobis detector is expected to achieve higher AUROC than MSP, particularly for OOD scenarios where the model remains overconfident (i.e., where the feature representation diverges from training data even though the output logit does not). The gap is likely largest for the **fog** or **different-town** scenario, where the output layer confidence is partially preserved but the deep feature distribution has shifted. For the night scenario, the signal degradation is severe enough that both methods may perform well. The different-town case is the most interesting: MSP may fail entirely (model stays confident), while Mahalanobis may detect the structural feature shift because the feature vectors lie away from the training distribution manifold.

---

## Exercise 9.8: Extending the Safety Analysis for OOD

### 1. Hazard

Reviewing the hazards from Exercise 2.4, the existing list covers failures like "vehicle fails to brake for a pedestrian" and "vehicle fails to stop at a red light", attributed to model misprediction. However, there is no entry that specifically captures **the mechanism of OOD-induced failure combined with absent detection**. The hazard needs to be added:

| ID  | Hazard | Loss(es) | Likelihood | Severity |
|-----|--------|----------|------------|----------|
| H-5 | The perception system operates on a camera input that is outside the ODD and this is not detected, causing the planner to act on unreliable perception output | L-1 (pedestrian collision), L-2 (vehicle collision), L-3 (traffic violation) | High | High |

*Likelihood is rated High because OOD conditions (adverse weather, night, novel locations) are frequent in real deployment and the existing system has no OOD monitor. Severity is High because the planner receives confident but wrong outputs with no indication that anything is wrong.*

### 2. Unsafe Control Action

| ID | Controller | Control action | UCA type | Hazard(s) | Unsafe scenario |
|----|------------|----------------|----------|-----------|-----------------|
| UCA-OOD-1 | Planning module | Maintain speed / no braking command | Provided unsafely (action issued when it should not be) | H-5, H-1 | The planner continues at speed while the camera input is out-of-ODD (e.g., fog, night, or novel location) and the perception output is untrustworthy. The pedestrian detector outputs "no pedestrian" with high confidence on a foggy frame where a pedestrian is actually present, and no OOD flag is raised. The planner has no indication that the perception is unreliable and does not issue a braking command. |

*Link to hazard:* UCA-OOD-1 directly causes H-5 (operating on undetected OOD input with unreliable perception output) and, given worst-case environmental conditions, leads to H-1 (vehicle fails to brake for a pedestrian in its path).

### 3. Safety Constraints

**Model-level constraint (SC-OOD-M1):**

> The OOD monitor must achieve a true positive rate (recall) of ≥ 0.95 on fog, night, and other clearly out-of-ODD inputs at a false positive rate of ≤ 0.05 on in-distribution sunny/daytime inputs, as measured on a held-out evaluation set covering all known OOD scenarios.

*Justification:* The severity of H-5 is High — an undetected OOD input can lead directly to a collision with a pedestrian. The asymmetric threshold (very high recall, moderate precision) reflects that the cost of a missed OOD detection (false negative) vastly exceeds the cost of an unnecessary false alarm (false positive), which merely triggers a conservative fallback.

**System-level constraint (SC-OOD-S1):**

> When the OOD monitor flags the current camera input as out-of-distribution, the planning module must immediately transition to a safe-stop or minimum-risk-condition state: reduce vehicle speed to ≤ 10 km/h within 3 seconds, activate hazard signals, and request human operator takeover. The vehicle must not resume autonomous operation until the OOD flag is cleared by a qualified operator or by a confirmed return to in-distribution conditions.

*Justification:* Detecting an OOD input does not by itself make the vehicle safe — the system must also respond appropriately. A conservative speed reduction and mandatory human handover ensures that, even if perception is completely unreliable, the vehicle is in a state where the human operator can recover the situation.

### 4. Residual Risk

Even with a perfect OOD detector, significant residual risk remains for the following reasons:

1. **Detection latency.** An OOD flag can only be raised after the camera frame has been processed. There is an unavoidable delay between the onset of an OOD condition (e.g., sudden fog bank) and the detection and physical response of the vehicle. During this window, the vehicle continues on the basis of the last reliable perception output, which may already be wrong.

2. **The fallback behaviour itself carries risk.** Transitioning to a safe-stop state at speed is not instantaneous and not without hazard — emergency braking can cause a rear-end collision; stopping on a foggy highway is itself dangerous. Detecting OOD does not guarantee a safe outcome, only a safer one.

3. **OOD detection does not restore perception.** The UCA is eliminated (the planner no longer acts confidently on untrustworthy perception) but the underlying hazard — being in a dangerous environment without reliable perception — is not resolved. A perfect OOD detector converts silent failure into uncertain failure, but the vehicle still cannot perceive its surroundings. Only a sensor fusion system with OOD-robust redundancy would address the hazard more fully.

4. **In-distribution failures are unaffected.** The OOD monitor cannot help when the model fails on an in-distribution input (e.g., a partially occluded pedestrian in good weather). The OOD monitor and the perception model are orthogonal components: improving OOD detection does not improve the model's base accuracy on its ODD.

5. **ODD boundary is imprecise.** Our revised ODD includes different CARLA towns as in-distribution. If the model underperforms on those towns (as the experiments may show), the OOD monitor will not flag those inputs, and the planner will again act on unreliable perception with no warning. The ODD definition itself carries residual risk when model performance has not been validated across the entire declared ODD.

**Conclusion:** Detecting OOD fully addresses UCA-OOD-1 (the planner no longer continues at speed on untrustworthy perception without a signal), but does not fully resolve H-5. The hazard is mitigated — a conservative fallback reduces the probability that OOD-induced misperception leads to a loss — but it is not eliminated. Eliminating H-5 would require either making the perception system robust to OOD inputs (through domain adaptation, ensemble methods, or sensor redundancy) or ensuring the vehicle never enters OOD conditions (geofencing, weather monitoring, operational restrictions).
