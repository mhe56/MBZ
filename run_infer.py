import torch
import numpy as np
from lib.model.model_action import ActionNet
from lib.model.DSTformer import DSTformer  # This is your backbone
from yacs.config import CfgNode as CN

# 1. Build config
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

# 2. Initialize model correctly with unpacked config
# Create backbone
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

# Wrap with classification head
model = ActionNet(
    backbone=backbone,
    dim_rep=512,
    num_classes=cfg.MODEL.NUM_CLASSES,
    num_joints=cfg.MODEL.NUM_JOINTS,
    version='class'
)


# 3. Load pretrained weights
# Load checkpoint
checkpoint = torch.load("checkpoints/action/MotionBERT_ft_NTU60_xsub.pth", map_location='cpu', weights_only=True)

# Strip "module.backbone." prefix from all keys
state_dict = checkpoint["model"]
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module.backbone."):
        new_k = k.replace("module.backbone.", "")
        new_state_dict[new_k] = v
    else:
        # Optionally print unknown keys
        print("Skipping unexpected key:", k)

# Load cleaned state_dict
model.load_state_dict(new_state_dict, strict=False)

model.eval()

# 4. Load input (dummy or real ZED2i .npy input)
data = np.load("custom_input.npy")  # Shape: [1, T, 17, 3]
tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Shape: [1, 1, 60, 17, 3]



# 5. Inference
# Run prediction
with torch.no_grad():
    output = model(tensor)

    # Handle different output shapes
    if output.ndim == 3:
        # If output is [B, T, C] → average over time (T)
        output = output.mean(dim=1)
    elif output.ndim == 2:
        # Already [B, C] → do nothing
        pass
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    # Now it's [B, C] — get predicted class
    predicted_class = output.argmax(dim=1).item()



# 6. Action labels (NTU60)
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

# 7. Print result
print(f"\n✅ Predicted Action: {ntu_actions[predicted_class]} (Class ID: {predicted_class})")
