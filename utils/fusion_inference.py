"""Fusion Inference Logic

Goal:
  Combine heterogeneous modality outputs (action, emotion, audio, pose, objects)
  into a unified event classification with a risk tier and explanatory reasons.

Inputs:
  outputs: Dict with optional sub-dicts. Expected (but optional) keys:
    action: {label: prob, ...} OR { 'fight':0.9, 'theft':0.2 }
    emotion: {'Angry':0.7, 'Fear':0.1, ...}
    audio: {'gunshot':0.2, 'scream':0.6, ...}
    pose: {'aggressive':0.4, 'trespass':0.3}
    objects: list[ {class_id:int, confidence:float, bbox:[...]} ]

Configurable:
  - Weights via Config.FUSION_WEIGHTS
  - Thresholds via env (defaults below)

Outputs:
  {
    'label': str,              # chosen dominant event label
    'risk': 'low'|'medium'|'high',
    'confidence': float,       # aggregated score for chosen label
    'scores': {label:score,...},
    'reasons': list[(source,str,float)],  # (modality,label,contribution)
    'raw': original_outputs_subset
  }
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os
from config.config import Config


LOW_THRESHOLD = float(os.getenv('FUSION_LOW_THRESHOLD', '0.3'))
HIGH_THRESHOLD = float(os.getenv('FUSION_HIGH_THRESHOLD', '0.7'))

# Mapping sets for categorical escalation
HIGH_PRIORITY_ACTION = {'fire', 'explosion', 'fight', 'collapse'}
HIGH_PRIORITY_AUDIO = {'gunshot', 'explosion', 'scream'}
MED_PRIORITY_ACTION = {'theft', 'running'}
MED_PRIORITY_AUDIO = {'glass_break', 'siren'}
EMOTION_ESCALATE = {'Angry', 'Fear'}

def _top_prob(d: Dict[str, float]) -> Tuple[str, float]:
    if not d:
        return ('', 0.0)
    k = max(d, key=d.get)
    return k, float(d[k])

def _collect_probability_map(modality_data: Any) -> Dict[str, float]:
    if isinstance(modality_data, dict):
        return {k: float(v) for k, v in modality_data.items() if isinstance(v, (int, float))}
    return {}

def fuse_modalities(outputs: Dict[str, Any]) -> Dict[str, Any]:
    weights = Config.FUSION_WEIGHTS
    reasons: List[Tuple[str, str, float]] = []

    action_probs = _collect_probability_map(outputs.get('action', {}))
    emotion_probs = _collect_probability_map(outputs.get('emotion', {}))
    audio_probs = _collect_probability_map(outputs.get('audio', {}))
    pose_probs = _collect_probability_map(outputs.get('pose', {}))

    # Scores for candidate composite events
    scores: Dict[str, float] = {}

    # Violence composite example
    fight_p = action_probs.get('fight', 0.0)
    aggressive_pose = pose_probs.get('aggressive', 0.0)
    scream_audio = audio_probs.get('scream', 0.0)
    violence_score = (
        weights['action'] * fight_p +
        weights['pose'] * aggressive_pose +
        weights['audio'] * scream_audio
    )
    if violence_score > 0:
        scores['violence'] = violence_score
        if fight_p: reasons.append(('action', 'fight', weights['action'] * fight_p))
        if aggressive_pose: reasons.append(('pose', 'aggressive', weights['pose'] * aggressive_pose))
        if scream_audio: reasons.append(('audio', 'scream', weights['audio'] * scream_audio))

    # Intrusion composite
    trespass_pose = pose_probs.get('trespass', 0.0)
    running_action = action_probs.get('running', 0.0)
    intrusion_score = (
        weights['pose'] * trespass_pose +
        weights['action'] * running_action
    )
    if intrusion_score > 0:
        scores['intrusion'] = intrusion_score
        if trespass_pose: reasons.append(('pose', 'trespass', weights['pose'] * trespass_pose))
        if running_action: reasons.append(('action', 'running', weights['action'] * running_action))

    # Fire detection composite
    fire_action = action_probs.get('fire', 0.0)
    explosion_audio = audio_probs.get('explosion', 0.0)
    fire_score = weights['action'] * fire_action + weights['audio'] * explosion_audio
    if fire_score > 0:
        scores['fire'] = fire_score
        if fire_action: reasons.append(('action', 'fire', weights['action'] * fire_action))
        if explosion_audio: reasons.append(('audio', 'explosion', weights['audio'] * explosion_audio))

    # Fallback: if no composite scores, use top action/audio as label
    if not scores:
        a_label, a_p = _top_prob(action_probs)
        au_label, au_p = _top_prob(audio_probs)
        em_label, em_p = _top_prob(emotion_probs)
        cand = [(a_label, a_p), (au_label, au_p), (em_label, em_p)]
        cand = [c for c in cand if c[0]]
        if not cand:
            return {
                'label': 'none', 'risk': 'low', 'confidence': 0.0, 'scores': {}, 'reasons': [], 'raw': {}
            }
        label, conf = max(cand, key=lambda x: x[1])
        base_risk = 'high' if label in HIGH_PRIORITY_ACTION.union(HIGH_PRIORITY_AUDIO) else ('medium' if conf > 0.5 else 'low')
        return {
            'label': label,
            'risk': base_risk,
            'confidence': conf,
            'scores': {label: conf},
            'reasons': [('direct', label, conf)],
            'raw': {'action': action_probs, 'audio': audio_probs, 'emotion': emotion_probs, 'pose': pose_probs}
        }

    # Choose best label
    label = max(scores, key=scores.get)
    confidence = scores[label]

    # Escalation modifiers
    # Add extra weight if any high priority labels appear strongly
    escalation_bonus = 0.0
    for k, v in action_probs.items():
        if k in HIGH_PRIORITY_ACTION and v > 0.5:
            escalation_bonus += 0.5 * v
    for k, v in audio_probs.items():
        if k in HIGH_PRIORITY_AUDIO and v > 0.5:
            escalation_bonus += 0.5 * v
    # Emotion slight escalation
    for k, v in emotion_probs.items():
        if k in EMOTION_ESCALATE and v > 0.5:
            escalation_bonus += 0.2 * v
    confidence += escalation_bonus
    if escalation_bonus:
        reasons.append(('escalation', 'bonus', escalation_bonus))

    # Determine risk tier
    if confidence >= HIGH_THRESHOLD:
        risk = 'high'
    elif confidence >= LOW_THRESHOLD:
        risk = 'medium'
    else:
        risk = 'low'

    return {
        'label': label,
        'risk': risk,
        'confidence': float(confidence),
        'scores': scores,
        'reasons': reasons,
        'raw': {
            'action': action_probs,
            'emotion': emotion_probs,
            'audio': audio_probs,
            'pose': pose_probs,
        }
    }
