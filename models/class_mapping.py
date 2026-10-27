# YOLO Model Class Mapping
# Class names expected by optional custom YOLO weights. The checkpoint and
# verifiable evaluation report are not included with this repository.

# Class ID to Event Name mapping
# Note: Model internally uses class_0, class_1, etc.
# This maps them to meaningful event names
CLASS_MAPPING = {
    0: 'explosion',
    1: 'fire',
    2: 'smoke',
    3: 'vehicle_accident',
}

# Reverse mapping (Event name to Class ID)
EVENT_TO_CLASS = {
    'explosion': 0,
    'fire': 1,
    'smoke': 2,
    'vehicle_accident': 3,
}

# Display names (user-friendly)
DISPLAY_NAMES = {
    0: 'Explosion',
    1: 'Fire',
    2: 'Smoke',
    3: 'Vehicle Accident'
}

# Risk levels for each event
RISK_LEVELS = {
    'explosion': 'CRITICAL',
    'smoke': 'HIGH',
    'fire': 'CRITICAL',
    'vehicle_accident': 'HIGH'
}

# Provisional thresholds; calibrate against a labeled validation set.
CONFIDENCE_THRESHOLDS = {
    'explosion': 0.25,
    'fire': 0.30,
    'smoke': 0.30,
    'vehicle_accident': 0.35
}

# Function to get event name from class ID
def get_event_name(class_id):
    """Convert class ID to event name"""
    return CLASS_MAPPING.get(class_id, f'unknown_class_{class_id}')

# Function to get display name from class ID  
def get_display_name(class_id):
    """Convert class ID to display name"""
    return DISPLAY_NAMES.get(class_id, f'Unknown Class {class_id}')
