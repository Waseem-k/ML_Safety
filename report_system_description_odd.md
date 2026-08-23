## 2. System Description

### 2.1 Provided system description

*(Reproduced verbatim from the assignment; not to be edited.)*

An autonomous vehicle (AV) prototype operates in an urban environment. The system is composed of the following components:

1. **Camera:** A single forward-facing RGB camera, rigidly mounted behind the windshield, provides images at 10 Hz. There are no other perception sensors (no radar, no LiDAR, no ultrasonic).
2. **Perception models:** Three separate binary classifiers each receive the camera frame and output one binary prediction: *pedestrian present*, *vehicle present*, or *traffic light present*. Each model was trained exclusively on sunny daytime data collected in the CARLA simulator.
3. **Distance oracle:** A separate model estimates the distance to detected pedestrians and vehicles. This model may be assumed perfect for the sake of this course.
4. **Planning module:** The rule-based planner consumes the three perception outputs together with the current vehicle speed and steering angle. Its primary safety-relevant decisions are whether to continue driving, request deceleration, or issue an emergency brake command. Emergency braking is triggered when the pedestrian detector reports a pedestrian within a critical distance, when the vehicle detector indicates a road user ahead within a speed-dependent threshold, or when the traffic-light detector indicates that stopping is required at an approaching intersection. The planner does not have access to raw camera images and relies entirely on the perception models' outputs.
5. **Vehicle actuators:** Throttle, brake, and steering commands are executed by drive-by-wire actuators. Emergency braking applies maximum deceleration within the physical limits of the vehicle.
6. **Human safety operator:** A trained operator sits in the driver seat and monitors the road. The operator can override the autopilot at any time by pressing the brake pedal or turning the steering wheel. The operator receives a visual dashboard showing the current perception output and system status, but no auditory alerts are provided. During testing, operators work 4-hour shifts.
7. **Operating environment:** The vehicle is tested on public urban roads at speeds up to 50 km/h. The intended conditions are daytime, dry weather, and mapped intersections. However, weather and lighting conditions can change during a test drive (e.g. sudden cloud cover, low sun angle, rain onset).

### 2.2 Key design limitations (by construction)

- No sensor redundancy — a single camera is the only perception input.
- No other safety mechanisms are deployed yet.
- The human operator is the only fallback if the automation fails.
- The operator may exhibit a non-zero probability of delayed or missed intervention, especially under prolonged monitoring.

The traffic-light detector reports only the *presence* of a traffic light, not its *state* (red/green). The planner therefore cannot, on its own, decide whether to stop at a signalised intersection. This is a limitation of the system as specified that no model metric can close; it is carried forward unresolved to Section 6 as a known design limitation.

### 2.3 Architecture & Training

Three independent binary classifiers were trained — one per perception task (`has_pedestrian`, `has_traffic_light`, `has_vehicle`) — using the CARLA dataset's `train/train` split (7,200 labelled frames, 500×500 px RGB). Each frame is resized to 224×224 and normalised with ImageNet statistics prior to inference.

Two independent training approaches were implemented per task, yielding six checkpoints in total; the safety case in Section 5 is evidenced by the **ResNet-18 baseline**, with the EfficientNet-B3 variant retained as a comparison point for class-imbalance mitigation.

**Approach 1 — ResNet-18 baseline (primary, used for all verifications):**

| Element | Configuration |
|---|---|
| Backbone | ResNet-18, ImageNet-pretrained (`torchvision`), final FC layer replaced with a 2-way head |
| Loss | Cross-entropy (`BCEWithLogitsLoss` with auto-computed `pos_weight` in the balanced variant) |
| Optimiser | Adam, learning rate 1×10⁻⁴, cosine-annealing schedule |
| Training regime | Two-phase: 5 epochs with frozen backbone, followed by 20 epochs of full fine-tuning |
| Batch size | 16 |
| Train / test split | 7,200 / 3,600 images (standard test split); three additional held-out splits of 3,600 images each for fog, night, and unseen-town domain shift |
| Feature representation | 512-dimensional penultimate-layer (`avgpool`) activations, used for Mahalanobis-distance OOD scoring in Exercise 9 |

All three task-specific models share this architecture and training recipe; only the target label column differs.

**Approach 2 — EfficientNet-B3 (secondary, imbalance-mitigation comparison):** EfficientNet-B3 backbone with focal loss (γ=2.0; α=0.25, α=0.75 for the vehicle task), class-balanced batch sampling, MixUp/RandAugment augmentation, and progressive unfreezing over 10 epochs. This approach was evaluated for robustness to class imbalance but is not the basis for the quantitative claims in Section 5.

**Class imbalance in the training data**, which directly affects the interpretation of recall in Section 5: `has_vehicle` is near-always positive; `has_pedestrian` (1,718/7,200 positive, ≈24%) and `has_traffic_light` are minority-positive classes. This imbalance is the primary driver of the low pedestrian recall observed under the ResNet-18 baseline and motivates the pos-weighted loss and the EfficientNet-B3 comparison.

---

## 3. Operational Design Domain

| Dimension | Operating conditions (valid) | Non-operating conditions (invalid) | Detection |
|---|---|---|---|
| Weather | Dry, clear conditions as recorded in CARLA | Fog, rain onset, sudden cloud cover | Feature-space (Mahalanobis) or confidence-based anomaly monitor on the camera feed |
| Lighting | Sunny, daytime | Nighttime, low sun angle | Same monitor; image brightness/exposure statistics as a supporting signal |
| Camera | Forward-facing, rigidly mounted, unobstructed 10 Hz feed | Obscured, detached, or blocked lens | Static-frame / black-frame detection on the feed |
| Scene type | Urban CARLA road environment, any mapped town, under nominal weather/lighting (Town02, training town, and Town01 both included — see rationale below) | Non-urban, non-CARLA, or unmapped environments | GPS/map cross-reference |
| Vehicle speed | ≤ 50 km/h | > 50 km/h | Speed telemetry threshold |

**Scene-type rationale:** the training data was collected exclusively in one CARLA town, but the ODD is deliberately defined to include *any* CARLA urban town under nominal weather and lighting, not only the training town. Restricting the ODD to a single geographic layout would make the system practically undeployable, and the fog/night dimensions already capture the genuine sensor-degradation cases that warrant exclusion. Consequently, degraded performance on an unseen town (Town01) is treated as a **model capability gap** to be closed by better training-data coverage, not as evidence that the input should be rejected by the OOD monitor.

### 3.1 ODD Gap Analysis

The training data covers only one point in the weather, lighting, and scene-layout dimensions: dry, sunny, daytime frames from a single town. No training examples exist for fog, night, or any other CARLA town. Consequently:

- **No performance claim can be made a priori** for fog, night, or unseen-town inputs; Section 5 evaluates these gaps empirically rather than assuming coverage.
- The **speed** and **camera-condition** dimensions are structurally enforced (simulator speed cap, fixed rigid mount) and require no empirical coverage argument.

### 3.2 ODD Coverage

Evidence (*k*-projection coverage over the operating-condition combinations sampled by the training data, computed on the 3,600-image in-distribution test split):

| *k* | Coverage | Covered | Total |
|---|---|---|---|
| 1 | 1.0000 | 18 | 18 |
| 2 | 0.9710 | 134 | 138 |
| 3 | 0.8853 | 517 | 584 |

**Interpretation:** coverage decreases monotonically as *k* increases, from complete coverage of every individual condition value (*k*=1) to 88.5% coverage of all 3-way condition combinations. This is the expected combinatorial effect: pairwise and higher-order interactions between environmental factors are sparser in the training distribution than any single factor in isolation. The residual gap at *k*=3 (11.5% of combinations, 67 of 584) identifies specific condition combinations for which no direct training coverage exists, even though each individual condition is represented — these combinations are validity boundaries for any recall claim made in Section 5, not merely a hypothetical concern.

### 3.3 ODD Violation Response

The response to a detected ODD violation is a system-level safety control, discharged by design rather than by a single metric: on a monitor flag, the planning module reduces speed to ≤10 km/h within 3 seconds, activates hazard signals, and requests human takeover, resuming autonomous operation only once the flag clears. The ability to detect the violation is verified empirically in V-4; the adequacy of the response itself is argued in V-5.
