import torch

SCALES = 13, 26, 52
ANCHORS = torch.Tensor(
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)]
    + [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)]
    + [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
)

CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

COLORS = (
    "#FF3838",
    "#2C99A8",
    "#FF701F",
    "#6473FF",
    "#CFD231",
    "#48F90A",
    "#92CC17",
    "#3DDB86",
    "#1A9334",
    "#00D4BB",
    "#FF9D97",
    "#00C2FF",
    "#344593",
    "#FFB21D",
    "#0018EC",
    "#8438FF",
    "#520085",
    "#CB38FF",
    "#FF95C8",
    "#FF37C7",
)
