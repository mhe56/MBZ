import torch
import numpy as np
from yacs.config import CfgNode as CN
from lib.model.DSTformer import DSTformer
from lib.model.model_action import ActionNet  # Uses ActionNet instead of ActionModel

# 1. CONFIGURATION
cfg = CN()
cfg.MODEL = CN()
cfg.MODEL.NUM_JOINTS = 17
cfg.MODEL.NUM_CLASSES = 60
cfg.MODEL.DIM_IN = 3
cfg.MODEL.DIM_FEAT = 512
cfg.MODEL.DEPTH = 4
cfg.MODEL.NUM_HEADS = 8
cfg.MODEL.MLP_RATIO = 2.0
cfg.MODEL.DROP_RATE = 0.1
cfg.MODEL.ATTN_DROP_RATE = 0.1
cfg.MODEL.DROP_PATH_RATE = 0.1

# 2. KEYPOINT MAPPING: ZED2i (18) → H36M (17)
zed_to_h36m = [
    8,   # Hip (RHip)
    8,   # RHip
    9,   # RKnee
    10,  # RAnkle
    11,  # LHip
    12,  # LKnee
    13,  # LAnkle
    1,   # Spine ≈ Neck
    1,   # Neck
    0,   # Head ≈ Nose
    5,   # LShoulder
    6,   # LElbow
    7,   # LWrist
    2,   # RShoulder
    3,   # RElbow
    4,   # RWrist
    1    # Chest ≈ Neck
]

# 3. MODEL SETUP
backbone = DSTformer(
    dim_in=cfg.MODEL.DIM_IN,
    dim_out=cfg.MODEL.DIM_IN,
    dim_feat=cfg.MODEL.DIM_FEAT,
    dim_rep=512,
    depth=cfg.MODEL.DEPTH,
    num_heads=cfg.MODEL.NUM_HEADS,
    mlp_ratio=cfg.MODEL.MLP_RATIO,
    num_joints=cfg.MODEL.NUM_JOINTS,
    drop_rate=cfg.MODEL.DROP_RATE,
    attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
    drop_path_rate=cfg.MODEL.DROP_PATH_RATE
)

model = ActionNet(
    backbone=backbone,
    dim_rep=512,
    num_classes=cfg.MODEL.NUM_CLASSES,
    num_joints=cfg.MODEL.NUM_JOINTS,
    version='class'
)

# 4. LOAD CHECKPOINT
checkpoint = torch.load("checkpoints/action/MotionBERT_ft_NTU60_xsub.pth", map_location='cpu')
state_dict = checkpoint["model"]
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module.backbone."):
        new_state_dict[k.replace("module.backbone.", "")] = v
    else:
        print("Skipping unexpected key:", k)

model.load_state_dict(new_state_dict, strict=False)
model.eval()

# 5. LOAD ZED2i INPUT & CONVERT
zed_input = np.load("dummy_zed2i.npy")  # <- replace with your actual ZED2i data file
h36m_input = zed_input[:, zed_to_h36m, :]       # shape: [60, 17, 3]
final_input = np.expand_dims(h36m_input, axis=0)  # shape: [1, 60, 17, 3]
tensor = torch.tensor(final_input, dtype=torch.float32).unsqueeze(1)  # shape: [1, 1, 60, 17, 3]

# 6. INFERENCE
with torch.no_grad():
    output = model(tensor)
    if output.ndim == 3:
        output = output.mean(dim=1)
    elif output.ndim != 2:
        raise ValueError(f"Unexpected output shape: {output.shape}")
    predicted_class = output.argmax(dim=1).item()

# 7. ACTION LABELS
ntu_actions = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat", "take off a hat", "cheer up", "hand waving", "kicking something",
    "put something inside pocket", "taking something out of pocket",
    "hopping (one foot jumping)", "jump up", "make a phone call", "playing with phone",
    "typing on a keyboard", "pointing to something", "taking a selfie",
    "check time (from watch)", "rub two hands", "nod head/bow",
    "shake head", "wipe face", "salute", "put the palms together",
    "cross hands in front", "sneeze/cough", "staggering", "falling",
    "touch head (headache)", "touch chest (chest pain)", "touch back (backache)",
    "touch neck (neck ache)", "nausea or vomiting", "use a fan", "punching/slapping",
    "kicking other person", "pushing other person", "pat on back of other person",
    "point finger at other person", "hugging other person", "giving something",
    "handshaking", "walking towards each other", "walking apart from each other"
]

# 8. SHOW RESULT
print(f"\n✅ Predicted Action: {ntu_actions[predicted_class]} (Class ID: {predicted_class})")
