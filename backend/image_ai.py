# =============================================================================
# backend/image_ai.py
# Autonomous Disaster Response - Infrastructure Damage AI
#
# EXISTING APPLICATION INTERFACE IS PRESERVED:
#
#     run_accurate_ai_inference(image_path)
#
# RETURNS:
#
#     category, severity, annotated_path
#
#
# PIPELINE
#
#                 IMAGE
#                   |
#                   v
#              YOLOE-PF
#                   |
#          +--------+--------+
#          |                 |
#   Infrastructure       Other objects
#          |                 |
#          v                 |
#    Infrastructure         |
#       crops               |
#          |                 |
#          v                 |
#       CLIP DAMAGE         |
#          |                 |
#          +--------+--------+
#                   |
#                   v
#             SEVERITY ENGINE
#                   |
#          +--------+--------+
#          |        |        |
#        GREEN    YELLOW     RED
#
#
# FALLBACK:
#
# If YOLOE misses infrastructure:
#
#       IMAGE
#         |
#         v
#   CLIP SCENE CHECK
#         |
#    Infrastructure?
#         |
#        YES
#         |
#         v
#   Full-image damage analysis
#
# This prevents a collapsed bridge from becoming:
#
#       "No Infrastructure Damage Detected"
#       0%
#
# =============================================================================


import os
import re
import time
from functools import lru_cache

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

from ultralytics import YOLOE

from transformers import CLIPModel, AutoProcessor


# =============================================================================
# CONFIGURATION
# =============================================================================

# -----------------------------------------------------------------------------
# YOLOE
# -----------------------------------------------------------------------------
#
# IMPORTANT:
#
# We intentionally use the PROMPT-FREE checkpoint.
#
# DO NOT use:
#
#     yoloe-26s-seg.pt
#
# here because that requires YOLOE text prompting and therefore MobileCLIP.
#
# Prompt-free YOLOE uses its built-in vocabulary and does NOT require:
#
#     clip
#     mobileclip2_b.ts
#     git
#
# -----------------------------------------------------------------------------

YOLOE_MODEL_NAME = "yoloe-26s-seg-pf.pt"


# -----------------------------------------------------------------------------
# CLIP
# -----------------------------------------------------------------------------

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


# -----------------------------------------------------------------------------
# DEVICE
# -----------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# YOLOE DETECTION CONFIGURATION
# =============================================================================

# Lower than the previous 0.30 so that infrastructure is less likely
# to be missed.
YOLO_CONFIDENCE = 0.20


# Ignore extremely tiny detections.
MIN_INFRA_AREA_RATIO = 0.001


# =============================================================================
# INFRASTRUCTURE KEYWORDS
# =============================================================================
#
# YOLOE-PF has a very large vocabulary.
#
# We do NOT need to enumerate all 4585 classes.
#
# Instead, we inspect the returned class name and determine whether
# it belongs to infrastructure.
# =============================================================================

INFRASTRUCTURE_KEYWORDS = (
    "road",
    "roadway",
    "highway",
    "street",
    "pavement",
    "bridge",
    "overpass",
    "viaduct",
    "building",
    "house",
    "structure",
    "wall",
    "concrete",
    "masonry",
    "railway",
    "railroad",
    "rail",
    "track",
    "pole",
    "utility",
    "electric",
    "power line",
    "dam",
    "tunnel",
    "culvert",
    "drain",
    "drainage",
    "pipeline",
    "retaining wall",
    "tower",
    "footpath",
    "sidewalk",
    "pavement",
    "roof",
    "water tank",
    "industrial",
)


# =============================================================================
# NON-INFRASTRUCTURE KEYWORDS
# =============================================================================
#
# These objects are NEVER allowed to directly increase infrastructure
# severity.
# =============================================================================

NON_INFRASTRUCTURE_KEYWORDS = (
    "person",
    "human",
    "child",
    "man",
    "woman",
    "people",
    "animal",
    "elephant",
    "dog",
    "cat",
    "cow",
    "horse",
    "bird",
    "vehicle",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "motorbike",
    "bicycle",
    "boat",
    "tree",
    "plant",
    "vegetation",
    "grass",
    "forest",
    "rock",
    "stone",
    "cloud",
    "sky",
)


# =============================================================================
# DAMAGE CONDITION PROMPTS
# =============================================================================
#
# Multiple prompts per condition are used instead of one sentence.
#
# This makes the zero-shot classification less dependent on one particular
# wording.
# =============================================================================

DAMAGE_GROUPS = {

    "intact": [
        "a photo of intact undamaged infrastructure",
        "a photo of a normal undamaged road bridge or building",
        "a photo of infrastructure in good condition with no visible damage",
        "a photo of safe functioning infrastructure",
    ],

    "minor": [
        "a photo of infrastructure with minor surface damage",
        "a photo of infrastructure with small cracks or minor defects",
        "a photo of slightly damaged infrastructure",
        "a photo of infrastructure with cosmetic or minor damage",
    ],

    "moderate": [
        "a photo of infrastructure with moderate visible damage",
        "a photo of infrastructure with significant cracks or broken sections",
        "a photo of a damaged road bridge or building that is not collapsed",
        "a photo of infrastructure with partial structural damage",
    ],

    "severe": [
        "a photo of severely damaged infrastructure",
        "a photo of infrastructure with major structural failure",
        "a photo of infrastructure with severe cracks deformation and broken parts",
        "a photo of a severely damaged road bridge or building",
    ],

    "collapsed": [
        "a photo of collapsed infrastructure",
        "a photo of a completely destroyed bridge building or road structure",
        "a photo showing catastrophic structural collapse",
        "a photo of collapsed infrastructure surrounded by rubble and debris",
    ],
}


# =============================================================================
# DAMAGE TYPE PROMPTS
# =============================================================================

DAMAGE_TYPES = {

    "cracks": [
        "visible structural cracks in concrete or masonry",
        "large cracks in an infrastructure structure",
        "cracked concrete infrastructure",
    ],

    "broken wall": [
        "a broken structural wall",
        "a wall with major broken sections",
        "a damaged building wall",
    ],

    "collapse": [
        "a collapsed infrastructure structure",
        "a collapsed bridge or building",
        "structural collapse of infrastructure",
    ],

    "deformation": [
        "structural deformation of infrastructure",
        "severe bending or displacement of infrastructure",
        "a distorted bridge road or structure",
    ],

    "pothole / road break": [
        "a road with potholes",
        "a severely broken road surface",
        "damaged pavement with potholes",
    ],

    "debris / obstruction": [
        "infrastructure blocked by disaster debris",
        "a road blocked by rubble mud or debris",
        "a bridge obstructed by debris",
    ],

    "other visible damage": [
        "visible disaster damage to infrastructure",
        "damaged infrastructure with a structural defect",
        "visible damage to a civil infrastructure structure",
    ],
}


# =============================================================================
# INFRASTRUCTURE TYPE PROMPTS
# =============================================================================

INFRASTRUCTURE_TYPES = {

    "Bridge Infrastructure": [
        "a photo of a bridge",
        "a road bridge",
        "a damaged bridge",
        "a concrete bridge",
    ],

    "Road Infrastructure": [
        "a photo of a road",
        "a highway or roadway",
        "a paved road",
        "a damaged road",
    ],

    "Building / Structural Infrastructure": [
        "a building",
        "a house",
        "a concrete building",
        "a civil structure",
    ],

    "Railway Infrastructure": [
        "a railway track",
        "a railroad",
        "railway infrastructure",
    ],

    "Electrical Infrastructure": [
        "an electric pole",
        "a utility pole",
        "power infrastructure",
        "electric power lines",
    ],

    "Civil Infrastructure": [
        "a dam",
        "a tunnel",
        "a drainage structure",
        "a retaining wall",
        "a pipeline",
    ],
}


# =============================================================================
# SEVERITY BANDS
# =============================================================================
#
# The previous expected-value method could turn an intact image into
# 60%+ severity because probabilities were spread across several classes.
#
# This version uses the winning damage condition as the primary severity band.
# =============================================================================

SEVERITY_BANDS = {

    "intact": (
        5,
        25,
    ),

    "minor": (
        20,
        39,
    ),

    "moderate": (
        40,
        69,
    ),

    "severe": (
        70,
        89,
    ),

    "collapsed": (
        90,
        100,
    ),
}


DAMAGE_ORDER = [
    "intact",
    "minor",
    "moderate",
    "severe",
    "collapsed",
]


# =============================================================================
# BASIC HELPERS
# =============================================================================

def clamp(
    value,
    minimum,
    maximum,
):
    return max(
        minimum,
        min(
            float(value),
            maximum,
        ),
    )


def normalize_text(
    value,
):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).lower(),
    ).strip()


# =============================================================================
# ROBUST CLIP EMBEDDING EXTRACTION
# =============================================================================
#
# THIS FIXES YOUR:
#
# AttributeError:
# 'BaseModelOutputWithPooling' object has no attribute 'norm'
#
# Newer Transformers versions can return model-output objects from
# get_text_features/get_image_features.
#
# Instead of assuming the return value is already a Tensor, this code:
#
# 1. Uses CLIPModel.forward() when possible.
# 2. Extracts text_embeds/image_embeds.
# 3. Falls back to the underlying CLIP encoder + projection.
#
# =============================================================================

def _normalize_embedding(
    tensor,
):

    return F.normalize(
        tensor,
        p=2,
        dim=-1,
    )


# =============================================================================
# CLIP MODEL
# =============================================================================

@lru_cache(
    maxsize=1
)
def get_clip_model():

    print(
        "[AI] Loading CLIP..."
    )

    model = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME
    )

    processor = AutoProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "[AI] CLIP loaded."
    )

    return (
        model,
        processor,
    )


# =============================================================================
# TEXT EMBEDDING
# =============================================================================

def encode_text(
    texts,
):

    model, processor = get_clip_model()

    inputs = processor(
        text=list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        # -------------------------------------------------------------
        # Preferred modern path
        # -------------------------------------------------------------

        try:

            output = model(
                **inputs,
                return_dict=True,
            )

            text_embeds = getattr(
                output,
                "text_embeds",
                None,
            )

            if torch.is_tensor(
                text_embeds
            ):

                return _normalize_embedding(
                    text_embeds
                )

        except Exception:

            pass

        # -------------------------------------------------------------
        # Compatibility fallback
        # -------------------------------------------------------------

        text_output = model.text_model(
            **inputs,
            return_dict=True,
        )

        pooled = text_output.pooler_output

        projection = model.text_projection

        if callable(projection):

            features = projection(
                pooled
            )

        else:

            features = pooled @ projection

        return _normalize_embedding(
            features
        )


# =============================================================================
# IMAGE EMBEDDING
# =============================================================================

def encode_images(
    images,
):

    model, processor = get_clip_model()

    inputs = processor(
        images=list(images),
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(
        DEVICE
    )

    with torch.inference_mode():

        # -------------------------------------------------------------
        # Preferred modern path
        # -------------------------------------------------------------

        try:

            output = model(
                pixel_values=pixel_values,
                return_dict=True,
            )

            image_embeds = getattr(
                output,
                "image_embeds",
                None,
            )

            if torch.is_tensor(
                image_embeds
            ):

                return _normalize_embedding(
                    image_embeds
                )

        except Exception:

            pass

        # -------------------------------------------------------------
        # Compatibility fallback
        # -------------------------------------------------------------

        vision_output = model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )

        pooled = vision_output.pooler_output

        projection = model.visual_projection

        if callable(projection):

            features = projection(
                pooled
            )

        else:

            features = pooled @ projection

        return _normalize_embedding(
            features
        )


# =============================================================================
# CACHED DAMAGE TEXT FEATURES
# =============================================================================

@lru_cache(
    maxsize=1
)
def get_damage_text_features():

    prompts = []

    groups = []

    for group in DAMAGE_ORDER:

        for prompt in DAMAGE_GROUPS[
            group
        ]:

            prompts.append(
                prompt
            )

            groups.append(
                group
            )

    features = encode_text(
        prompts
    )

    return (
        features,
        tuple(groups),
    )


# =============================================================================
# CACHED DAMAGE TYPE FEATURES
# =============================================================================

@lru_cache(
    maxsize=1
)
def get_damage_type_text_features():

    prompts = []

    groups = []

    for group, prompt_list in DAMAGE_TYPES.items():

        for prompt in prompt_list:

            prompts.append(
                prompt
            )

            groups.append(
                group
            )

    features = encode_text(
        prompts
    )

    return (
        features,
        tuple(groups),
    )


# =============================================================================
# CACHED INFRASTRUCTURE TYPE FEATURES
# =============================================================================

@lru_cache(
    maxsize=1
)
def get_infrastructure_type_features():

    prompts = []

    groups = []

    for group, prompt_list in INFRASTRUCTURE_TYPES.items():

        for prompt in prompt_list:

            prompts.append(
                prompt
            )

            groups.append(
                group
            )

    features = encode_text(
        prompts
    )

    return (
        features,
        tuple(groups),
    )


# =============================================================================
# CACHED SCENE FEATURES
# =============================================================================

@lru_cache(
    maxsize=1
)
def get_scene_features():

    prompts = [

        # Infrastructure
        (
            "a photograph containing infrastructure "
            "such as a road bridge building or civil structure"
        ),

        # No infrastructure
        (
            "a photograph containing only people animals vehicles "
            "vegetation or natural scenery and no infrastructure"
        ),
    ]

    return encode_text(
        prompts
    )


# =============================================================================
# YOLOE PROMPT-FREE MODEL
# =============================================================================

@lru_cache(
    maxsize=1
)
def get_yoloe_model():

    print(
        "[AI] Loading YOLOE prompt-free model..."
    )

    model = YOLOE(
        YOLOE_MODEL_NAME
    )

    print(
        "[AI] YOLOE prompt-free model loaded."
    )

    return model


# =============================================================================
# CLASSIFICATION HELPERS
# =============================================================================

def is_infrastructure_label(
    label,
):

    text = normalize_text(
        label
    )

    # Never classify these as infrastructure.
    for keyword in NON_INFRASTRUCTURE_KEYWORDS:

        if keyword in text:

            return False

    for keyword in INFRASTRUCTURE_KEYWORDS:

        if keyword in text:

            return True

    return False


def is_non_infrastructure_label(
    label,
):

    text = normalize_text(
        label
    )

    for keyword in NON_INFRASTRUCTURE_KEYWORDS:

        if keyword in text:

            return True

    return False


# =============================================================================
# YOLOE INFRASTRUCTURE DETECTION
# =============================================================================

def detect_infrastructure(
    image_path,
):

    model = get_yoloe_model()

    results = model.predict(

        source=image_path,

        imgsz=640,

        conf=YOLO_CONFIDENCE,

        device=YOLO_DEVICE,

        verbose=False,

        max_det=50,
    )

    if not results:

        return (
            None,
            [],
            [],
        )

    result = results[0]

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    image_width, image_height = image.size

    infrastructure = []

    non_infrastructure = []

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return (
            result,
            infrastructure,
            non_infrastructure,
        )

    boxes = result.boxes

    xyxy = boxes.xyxy.cpu().numpy()

    confidences = boxes.conf.cpu().numpy()

    class_ids = (
        boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    # YOLOE-PF supplies the built-in vocabulary here.
    names = getattr(
        result,
        "names",
        getattr(
            model,
            "names",
            {},
        ),
    )

    for (
        box,
        confidence,
        class_id,
    ) in zip(
        xyxy,
        confidences,
        class_ids,
    ):

        # -------------------------------------------------------------
        # Get class label
        # -------------------------------------------------------------

        if isinstance(
            names,
            dict,
        ):

            label = str(
                names.get(
                    int(class_id),
                    class_id,
                )
            )

        elif isinstance(
            names,
            (
                list,
                tuple,
            ),
        ) and class_id < len(names):

            label = str(
                names[
                    class_id
                ]
            )

        else:

            label = str(
                class_id
            )

        # -------------------------------------------------------------
        # Bounding box
        # -------------------------------------------------------------

        x1, y1, x2, y2 = box

        x1 = int(
            max(
                0,
                min(
                    image_width - 1,
                    x1,
                ),
            )
        )

        y1 = int(
            max(
                0,
                min(
                    image_height - 1,
                    y1,
                ),
            )
        )

        x2 = int(
            max(
                x1 + 1,
                min(
                    image_width,
                    x2,
                ),
            )
        )

        y2 = int(
            max(
                y1 + 1,
                min(
                    image_height,
                    y2,
                ),
            )
        )

        width = max(
            1,
            x2 - x1,
        )

        height = max(
            1,
            y2 - y1,
        )

        area_ratio = (
            width * height
        ) / float(
            image_width
            * image_height
        )

        detection = {

            "label": label,

            "confidence": float(
                confidence
            ),

            "bbox": (
                x1,
                y1,
                x2,
                y2,
            ),

            "area_ratio": area_ratio,
        }

        # -------------------------------------------------------------
        # INFRASTRUCTURE
        # -------------------------------------------------------------

        if is_infrastructure_label(
            label
        ):

            if (
                area_ratio
                >= MIN_INFRA_AREA_RATIO
            ):

                infrastructure.append(
                    detection
                )

        # -------------------------------------------------------------
        # NON-INFRASTRUCTURE
        # -------------------------------------------------------------

        elif is_non_infrastructure_label(
            label
        ):

            non_infrastructure.append(
                detection
            )

        else:

            # Unknown YOLOE object.
            #
            # It does not contribute to infrastructure severity.
            non_infrastructure.append(
                detection
            )

    return (
        result,
        infrastructure,
        non_infrastructure,
    )


# =============================================================================
# CLIP SCENE ANALYSIS
# =============================================================================
#
# This is the FALLBACK that prevents:
#
# collapsed bridge
#      ->
# YOLOE missed bridge
#      ->
# 0%
#
# =============================================================================

def classify_scene(
    image,
):

    image_features = encode_images(
        [image]
    )

    text_features = get_scene_features()

    similarity = (
        image_features
        @ text_features.T
    )

    probabilities = torch.softmax(
        similarity * 8.0,
        dim=-1,
    )

    probabilities = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    infrastructure_score = float(
        probabilities[0]
    )

    non_infrastructure_score = float(
        probabilities[1]
    )

    return (
        infrastructure_score,
        non_infrastructure_score,
    )


# =============================================================================
# INFRASTRUCTURE TYPE CLASSIFICATION
# =============================================================================

def classify_infrastructure_type(
    image,
):

    image_features = encode_images(
        [image]
    )

    text_features, groups = (
        get_infrastructure_type_features()
    )

    similarity = (
        image_features
        @ text_features.T
    )

    scores = []

    unique_groups = list(
        INFRASTRUCTURE_TYPES.keys()
    )

    for group in unique_groups:

        indices = [
            index

            for index, name in enumerate(
                groups
            )

            if name == group
        ]

        score = similarity[
            0,
            indices
        ].mean()

        scores.append(
            float(score)
        )

    best_index = int(
        np.argmax(
            scores
        )
    )

    return (
        unique_groups[
            best_index
        ],
        scores[
            best_index
        ],
    )


# =============================================================================
# INFRASTRUCTURE CROPS
# =============================================================================

def create_infrastructure_crops(
    image,
    infrastructure,
):

    crops = []

    for detection in infrastructure:

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        width = max(
            1,
            x2 - x1,
        )

        height = max(
            1,
            y2 - y1,
        )

        # Context margin
        margin_x = int(
            width * 0.10
        )

        margin_y = int(
            height * 0.10
        )

        crop_x1 = max(
            0,
            x1 - margin_x,
        )

        crop_y1 = max(
            0,
            y1 - margin_y,
        )

        crop_x2 = min(
            image.width,
            x2 + margin_x,
        )

        crop_y2 = min(
            image.height,
            y2 + margin_y,
        )

        crop = image.crop(
            (
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2,
            )
        )

        crops.append(
            crop
        )

    return crops


# =============================================================================
# DAMAGE ANALYSIS
# =============================================================================

def classify_infrastructure_damage(
    crops,
):

    if not crops:

        return []

    # -------------------------------------------------------------
    # Image embeddings
    # -------------------------------------------------------------

    image_features = encode_images(
        crops
    )

    # -------------------------------------------------------------
    # Damage condition embeddings
    # -------------------------------------------------------------

    damage_text_features, damage_groups = (
        get_damage_text_features()
    )

    similarity = (
        image_features
        @ damage_text_features.T
    )

    # -------------------------------------------------------------
    # Average multiple prompts belonging to the same condition.
    # -------------------------------------------------------------

    group_scores = []

    for group in DAMAGE_ORDER:

        indices = [

            i

            for i, current_group
            in enumerate(
                damage_groups
            )

            if current_group == group
        ]

        score = similarity[
            :,
            indices
        ].mean(
            dim=1
        )

        group_scores.append(
            score
        )

    group_scores = torch.stack(
        group_scores,
        dim=1,
    )

    # -------------------------------------------------------------
    # Convert to probabilities.
    # -------------------------------------------------------------

    probabilities = torch.softmax(
        group_scores * 8.0,
        dim=-1,
    )

    # -------------------------------------------------------------
    # Damage TYPE analysis
    # -------------------------------------------------------------

    type_features, type_groups = (
        get_damage_type_text_features()
    )

    type_similarity = (
        image_features
        @ type_features.T
    )

    type_probability = torch.sigmoid(
        (
            type_similarity
            - 0.18
        )
        * 12.0
    )

    outputs = []

    for row_index in range(
        len(crops)
    ):

        probs = (
            probabilities[
                row_index
            ]
            .detach()
            .cpu()
            .numpy()
        )

        top_index = int(
            np.argmax(
                probs
            )
        )

        top_condition = (
            DAMAGE_ORDER[
                top_index
            ]
        )

        top_probability = float(
            probs[
                top_index
            ]
        )

        # ---------------------------------------------------------
        # Damage type
        # ---------------------------------------------------------

        type_values = (
            type_probability[
                row_index
            ]
            .detach()
            .cpu()
            .numpy()
        )

        type_results = []

        for index, group in enumerate(
            type_groups
        ):

            type_results.append(
                (
                    group,
                    float(
                        type_values[
                            index
                        ]
                    ),
                )
            )

        type_results.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        if type_results:

            best_damage_type = (
                type_results[0][0]
            )

            best_damage_type_confidence = (
                type_results[0][1]
            )

        else:

            best_damage_type = (
                "other visible damage"
            )

            best_damage_type_confidence = 0.0

        # ---------------------------------------------------------
        # Severity
        # ---------------------------------------------------------

        minimum, maximum = (
            SEVERITY_BANDS[
                top_condition
            ]
        )

        # ---------------------------------------------------------
        # INTACT
        #
        # Never allow an intact image to become 50-60%
        # simply because CLIP has uncertainty.
        # ---------------------------------------------------------

        if top_condition == "intact":

            uncertainty = clamp(
                (
                    0.50
                    - top_probability
                )
                / 0.30,
                0.0,
                1.0,
            )

            severity = int(
                round(
                    minimum
                    + uncertainty
                    * (
                        maximum
                        - minimum
                    )
                )
            )

            severity = min(
                severity,
                25,
            )

        else:

            confidence_factor = clamp(
                (
                    top_probability
                    - 0.20
                )
                / 0.55,
                0.0,
                1.0,
            )

            severity = int(
                round(
                    minimum
                    + confidence_factor
                    * (
                        maximum
                        - minimum
                    )
                )
            )

            # -----------------------------------------------------
            # Strong damage evidence
            # -----------------------------------------------------

            if (
                best_damage_type_confidence
                >= 0.65
                and top_condition
                in (
                    "severe",
                    "collapsed",
                )
            ):

                severity += 5

            severity = int(
                clamp(
                    severity,
                    0,
                    100,
                )
            )

        outputs.append(

            {
                "condition": top_condition,

                "severity": severity,

                "confidence": top_probability,

                "probabilities": probs.tolist(),

                "damage_type": (
                    best_damage_type
                ),

                "damage_type_confidence": (
                    best_damage_type_confidence
                ),
            }
        )

    return outputs


# =============================================================================
# FULL IMAGE DAMAGE ANALYSIS
# =============================================================================
#
# Used only when YOLOE fails to produce an infrastructure box.
# =============================================================================

def classify_full_image_damage(
    image,
):

    results = classify_infrastructure_damage(
        [image]
    )

    if results:

        return results[0]

    return {
        "condition": "intact",
        "severity": 5,
        "confidence": 0.0,
        "probabilities": [],
        "damage_type": (
            "other visible damage"
        ),
        "damage_type_confidence": 0.0,
    }


# =============================================================================
# FALLBACK DETECTION
# =============================================================================

def create_fallback_infrastructure(
    image,
    confidence,
    label,
):

    return [

        {
            "label": label,

            "confidence": float(
                confidence
            ),

            "bbox": (
                0,
                0,
                image.width,
                image.height,
            ),

            "area_ratio": 1.0,

            "fallback": True,
        }
    ]


# =============================================================================
# FINAL SEVERITY
# =============================================================================

def calculate_final_severity(
    infrastructure,
    damage_results,
):

    if not infrastructure:

        return 0

    if not damage_results:

        return 0

    scores = []

    weights = []

    for (
        infrastructure_item,
        damage,
    ) in zip(
        infrastructure,
        damage_results,
    ):

        damage_score = float(
            damage.get(
                "severity",
                0,
            )
        )

        detection_confidence = float(
            infrastructure_item.get(
                "confidence",
                0,
            )
        )

        area_ratio = float(
            infrastructure_item.get(
                "area_ratio",
                0,
            )
        )

        # ---------------------------------------------------------
        # FALLBACK
        # ---------------------------------------------------------

        if infrastructure_item.get(
            "fallback",
            False,
        ):

            weight = 1.0

        else:

            area_weight = clamp(
                area_ratio * 8.0,
                0.5,
                2.0,
            )

            weight = max(
                0.15,
                detection_confidence,
            ) * area_weight

        scores.append(
            damage_score
        )

        weights.append(
            weight
        )

    if sum(weights) <= 0:

        return 0

    severity = float(
        np.average(
            scores,
            weights=weights,
        )
    )

    return int(
        round(
            clamp(
                severity,
                0,
                100,
            )
        )
    )


# =============================================================================
# CATEGORY NORMALIZATION
# =============================================================================

def normalize_category(
    infrastructure,
):

    if not infrastructure:

        return (
            "No Infrastructure Damage Detected"
        )

    labels = [

        normalize_text(
            item["label"]
        )

        for item in infrastructure
    ]

    # -------------------------------------------------------------
    # Bridge
    # -------------------------------------------------------------

    if any(

        (
            "bridge" in label
            or "overpass" in label
            or "viaduct" in label
        )

        for label in labels
    ):

        return (
            "Bridge Infrastructure"
        )

    # -------------------------------------------------------------
    # Road
    # -------------------------------------------------------------

    if any(

        any(

            keyword in label

            for keyword in (
                "road",
                "highway",
                "street",
                "roadway",
                "pavement",
            )

        )

        for label in labels
    ):

        return (
            "Road Infrastructure"
        )

    # -------------------------------------------------------------
    # Building
    # -------------------------------------------------------------

    if any(

        any(

            keyword in label

            for keyword in (
                "building",
                "house",
                "structure",
                "wall",
                "roof",
            )

        )

        for label in labels
    ):

        return (
            "Building / Structural Infrastructure"
        )

    # -------------------------------------------------------------
    # Electrical
    # -------------------------------------------------------------

    if any(

        any(

            keyword in label

            for keyword in (
                "pole",
                "power",
                "utility",
                "electric",
            )

        )

        for label in labels
    ):

        return (
            "Electrical Infrastructure"
        )

    # -------------------------------------------------------------
    # Railway
    # -------------------------------------------------------------

    if any(

        (
            "rail" in label
            or "track" in label
        )

        for label in labels
    ):

        return (
            "Railway Infrastructure"
        )

    # -------------------------------------------------------------
    # Civil
    # -------------------------------------------------------------

    if any(

        any(

            keyword in label

            for keyword in (
                "dam",
                "tunnel",
                "pipeline",
                "culvert",
                "drain",
                "retaining",
            )

        )

        for label in labels
    ):

        return (
            "Civil Infrastructure"
        )

    return "Infrastructure"


# =============================================================================
# SEVERITY COLOR
# =============================================================================

def severity_to_color(
    severity,
):

    if severity >= 70:

        return "RED"

    if severity >= 40:

        return "YELLOW"

    return "GREEN"


# =============================================================================
# ANNOTATION
# =============================================================================

def annotate_result(
    image_path,
    infrastructure,
    non_infrastructure,
    damage_results,
    final_severity,
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        image
    )

    # -------------------------------------------------------------
    # Font
    # -------------------------------------------------------------

    try:

        font = ImageFont.truetype(
            "arial.ttf",
            max(
                14,
                image.width // 70,
            ),
        )

    except Exception:

        font = ImageFont.load_default()

    # -------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------

    for index, detection in enumerate(
        infrastructure
    ):

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        if (
            index
            < len(
                damage_results
            )
        ):

            damage = (
                damage_results[
                    index
                ]
            )

            condition = str(
                damage.get(
                    "condition",
                    "infrastructure",
                )
            )

            severity = int(
                damage.get(
                    "severity",
                    0,
                )
            )

            damage_type = str(
                damage.get(
                    "damage_type",
                    "",
                )
            )

        else:

            condition = (
                "infrastructure"
            )

            severity = 0

            damage_type = ""

        # ---------------------------------------------------------
        # Border
        # ---------------------------------------------------------

        if severity >= 70:

            outline = "red"

        elif severity >= 40:

            outline = "orange"

        else:

            outline = "green"

        # ---------------------------------------------------------
        # Bounding box
        # ---------------------------------------------------------

        draw.rectangle(

            [
                x1,
                y1,
                x2,
                y2,
            ],

            outline=outline,

            width=max(
                3,
                image.width // 350,
            ),
        )

        # ---------------------------------------------------------
        # Label
        # ---------------------------------------------------------

        label = (

            f"{detection['label']} | "
            f"{condition} | "
            f"{severity}%"
        )

        if (
            damage_type
            and condition != "intact"
        ):

            label += (
                f" | {damage_type}"
            )

        try:

            text_box = draw.textbbox(
                (
                    x1,
                    y1,
                ),
                label,
                font=font,
            )

            draw.rectangle(
                text_box,
                fill=outline,
            )

        except Exception:

            pass

        draw.text(

            (
                x1,
                y1,
            ),

            label,

            fill="white",

            font=font,
        )

    # -------------------------------------------------------------
    # Non-infrastructure
    #
    # BLUE = detected but ignored for severity.
    # -------------------------------------------------------------

    for detection in (
        non_infrastructure
    ):

        x1, y1, x2, y2 = (
            detection["bbox"]
        )

        draw.rectangle(

            [
                x1,
                y1,
                x2,
                y2,
            ],

            outline="blue",

            width=2,
        )

        label = (
            f"{detection['label']} "
            "(ignored)"
        )

        try:

            text_box = draw.textbbox(
                (
                    x1,
                    y1,
                ),
                label,
                font=font,
            )

            draw.rectangle(
                text_box,
                fill="blue",
            )

        except Exception:

            pass

        draw.text(

            (
                x1,
                y1,
            ),

            label,

            fill="white",

            font=font,
        )

    # -------------------------------------------------------------
    # Severity banner
    # -------------------------------------------------------------

    if final_severity >= 70:

        banner_color = "red"

    elif final_severity >= 40:

        banner_color = "orange"

    else:

        banner_color = "green"

    banner_height = max(
        40,
        int(
            image.height * 0.08
        ),
    )

    draw.rectangle(

        [
            0,
            0,
            image.width,
            banner_height,
        ],

        fill=banner_color,
    )

    if infrastructure:

        banner_text = (

            "AI INFRASTRUCTURE SEVERITY: "
            f"{final_severity}%"
        )

    else:

        banner_text = (
            "NO INFRASTRUCTURE DAMAGE DETECTED"
        )

    draw.text(

        (
            15,
            10,
        ),

        banner_text,

        fill="white",

        font=font,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    root, extension = os.path.splitext(
        image_path
    )

    annotated_path = (
        root
        + "_annotated"
        + extension
    )

    image.save(
        annotated_path,
        quality=92,
    )

    return annotated_path


# =============================================================================
# MAIN APPLICATION FUNCTION
# =============================================================================

def run_accurate_ai_inference(
    image_path,
):

    start_time = time.time()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # =========================================================================
    # LOAD IMAGE
    # =========================================================================

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # =========================================================================
    # STAGE 1
    #
    # YOLOE prompt-free infrastructure detection
    # =========================================================================

    (
        yolo_result,
        infrastructure,
        non_infrastructure,
    ) = detect_infrastructure(
        image_path
    )

    # =========================================================================
    # PRIMARY PATH
    #
    # Infrastructure detected.
    # =========================================================================

    if infrastructure:

        # -------------------------------------------------------------
        # Crop infrastructure only
        # -------------------------------------------------------------

        infrastructure_crops = (
            create_infrastructure_crops(
                image,
                infrastructure,
            )
        )

        # -------------------------------------------------------------
        # Damage analysis
        # -------------------------------------------------------------

        damage_results = (
            classify_infrastructure_damage(
                infrastructure_crops
            )
        )

        # -------------------------------------------------------------
        # Severity
        # -------------------------------------------------------------

        final_severity = (
            calculate_final_severity(
                infrastructure,
                damage_results,
            )
        )

        # -------------------------------------------------------------
        # Category
        # -------------------------------------------------------------

        category = (
            normalize_category(
                infrastructure
            )
        )

        # -------------------------------------------------------------
        # Annotation
        # -------------------------------------------------------------

        annotated_path = (
            annotate_result(
                image_path,
                infrastructure,
                non_infrastructure,
                damage_results,
                final_severity,
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[AI] "
            f"{os.path.basename(image_path)} "
            f"| infrastructure={len(infrastructure)} "
            f"| other={len(non_infrastructure)} "
            f"| severity={final_severity}% "
            f"| time={elapsed:.2f}s"
        )

        return (
            category,
            final_severity,
            annotated_path,
        )

    # =========================================================================
    # FALLBACK PATH
    #
    # YOLOE missed infrastructure.
    #
    # DO NOT immediately return 0%.
    #
    # This is what fixes your collapsed bridge example.
    # =========================================================================

    (
        infrastructure_score,
        non_infrastructure_score,
    ) = classify_scene(
        image
    )

    print(
        f"[AI] "
        f"YOLOE found no infrastructure. "
        f"CLIP scene infrastructure score="
        f"{infrastructure_score:.3f}"
    )

    # =========================================================================
    # FALLBACK THRESHOLD
    # =========================================================================
    #
    # 0.55+ = strong infrastructure evidence
    #
    # 0.45-0.55 = uncertain.
    #
    # We allow the damage classifier to rescue a clearly damaged
    # infrastructure scene in this uncertain region.
    # =========================================================================

    fallback_damage = (
        classify_full_image_damage(
            image
        )
    )

    damage_condition = (
        fallback_damage.get(
            "condition",
            "intact",
        )
    )

    damage_confidence = float(
        fallback_damage.get(
            "confidence",
            0.0,
        )
    )

    damage_type_confidence = float(
        fallback_damage.get(
            "damage_type_confidence",
            0.0,
        )
    )

    severe_condition = (
        damage_condition
        in (
            "moderate",
            "severe",
            "collapsed",
        )
    )

    strong_damage_evidence = (

        severe_condition

        and (
            damage_confidence
            >= 0.27
        )

    ) or (

        damage_type_confidence
        >= 0.68

        and damage_condition
        != "intact"
    )

    fallback_should_activate = (

        infrastructure_score
        >= 0.45

    ) or (

        infrastructure_score
        >= 0.38

        and strong_damage_evidence
    )

    # =========================================================================
    # FALLBACK INFRASTRUCTURE
    # =========================================================================

    if fallback_should_activate:

        # -------------------------------------------------------------
        # Identify infrastructure type using CLIP.
        # -------------------------------------------------------------

        fallback_category, _ = (
            classify_infrastructure_type(
                image
            )
        )

        # -------------------------------------------------------------
        # Full image becomes the infrastructure crop.
        #
        # This is intentional:
        #
        # YOLOE missed the object, so we allow the damage model to
        # inspect the entire image.
        # -------------------------------------------------------------

        fallback_infrastructure = (
            create_fallback_infrastructure(
                image,
                infrastructure_score,
                fallback_category,
            )
        )

        damage_results = [
            fallback_damage
        ]

        # -------------------------------------------------------------
        # Final severity
        # -------------------------------------------------------------

        final_severity = (
            calculate_final_severity(
                fallback_infrastructure,
                damage_results,
            )
        )

        # -------------------------------------------------------------
        # Annotation
        # -------------------------------------------------------------

        annotated_path = (
            annotate_result(
                image_path,
                fallback_infrastructure,
                non_infrastructure,
                damage_results,
                final_severity,
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[AI] "
            f"{os.path.basename(image_path)} "
            f"| FALLBACK "
            f"| type={fallback_category} "
            f"| condition={damage_condition} "
            f"| severity={final_severity}% "
            f"| time={elapsed:.2f}s"
        )

        return (
            fallback_category,
            final_severity,
            annotated_path,
        )

    # =========================================================================
    # TRUE NO-INFRASTRUCTURE CASE
    #
    # Example:
    #
    #       elephant
    #       dog
    #       people
    #       trees
    #
    # with no meaningful infrastructure.
    # =========================================================================

    annotated_path = (
        annotate_result(
            image_path,
            [],
            non_infrastructure,
            [],
            0,
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"[AI] "
        f"{os.path.basename(image_path)} "
        f"| NO INFRASTRUCTURE "
        f"| severity=0% "
        f"| time={elapsed:.2f}s"
    )

    return (
        "No Infrastructure Damage Detected",
        0,
        annotated_path,
    )