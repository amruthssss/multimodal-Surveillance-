# YOLO Model Class Mapping
# Based on your Kaggle training results

# Class ID to Event Name mapping
# Note: Model internally uses class_0, class_1, etc.
# This maps them to meaningful event names
CLASS_MAPPING = {
    0: 'explosion',      # class_0 → 99.5% mAP - Excellent!
    1: 'fighting',       # class_1 → 49.4% mAP - Needs more data  
    2: 'fire',           # class_2 → 89.5% mAP - Excellent!
    3: 'vehicle_accident' # class_3 → 61.8% mAP - Good
}

# Reverse mapping (Event name to Class ID)
EVENT_TO_CLASS = {
    'explosion': 0,
    'fighting': 1,
    'fire': 2,
    'vehicle_accident': 3
}

# Display names (user-friendly)
DISPLAY_NAMES = {
    0: 'Explosion',
    1: 'Fighting',
    2: 'Fire',
    3: 'Vehicle Accident'
}

# Risk levels for each event
RISK_LEVELS = {
    'explosion': 'CRITICAL',
    'fighting': 'HIGH',
    'fire': 'CRITICAL',
    'vehicle_accident': 'HIGH'
}

# Confidence thresholds (based on training performance)
CONFIDENCE_THRESHOLDS = {
    'explosion': 0.25,      # High accuracy, can use lower threshold
    'fighting': 0.40,       # Lower accuracy, use higher threshold  
    'fire': 0.30,           # High accuracy
    'vehicle_accident': 0.35 # Good accuracy
}

# Function to get event name from class ID
def get_event_name(class_id):
    """Convert class ID to event name"""
    return CLASS_MAPPING.get(class_id, f'unknown_class_{class_id}')

# Function to get display name from class ID  
def get_display_name(class_id):
    """Convert class ID to display name"""
    return DISPLAY_NAMES.get(class_id, f'Unknown Class {class_id}')
