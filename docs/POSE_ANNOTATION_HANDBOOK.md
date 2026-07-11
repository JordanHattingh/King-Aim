# King Aim Pose Annotation Handbook

King Aim uses one class, `human`, and four ordered midline keypoints: `head`, `neck`, `upper_chest`, `hip`. Never change the order between labels, YAML, manifests, exports, or the C# decoder.

## Label contract

Each row is `class cx cy width height` followed by four `x y visibility` triples. Values are normalized to the full image. Visibility is `0` when absent or not reasonably inferable, `1` when occluded but inferable, and `2` when clearly visible. Stock Ultralytics supervises coordinates for both `1` and `2`.

- **Head:** visible center of the head or helmet, excluding name tags and sights.
- **Neck:** base of the neck between the shoulders.
- **Upper chest:** sternum area, stable across armour and clothing.
- **Hip:** pelvis center, never an individual left or right hip.

## Scene rules

- Standing, crouching, prone, motion-blurred, smoke-obscured, edge-clipped, and unusual-armour players remain positive when a human target is identifiable.
- For wall occlusion, label the visible body box and mark inferable hidden keypoints `1`; use `0` when inference is unreliable.
- Head-only targets receive a tight visible box; hidden torso keypoints are `0` unless their position is genuinely inferable.
- Overlapping people receive separate boxes and keypoints. Never merge two bodies into one label.
- Friendly/enemy crossings remain `human`; semantic role is resolved outside the pose model.
- Dead bodies, viewmodel hands, mannequins, posters, HUD markers, and shadows are negative unless a future dataset policy explicitly changes their role.
- Downed living players are positive and use the same anatomical definitions.

## Review gate

The pilot set is double-reviewed. Box IoU below `0.85`, normalized keypoint disagreement above `0.05` of box diagonal, or any visibility disagreement requires adjudication. Run `training/tools/audit_pose_annotations.py` before every training run.
