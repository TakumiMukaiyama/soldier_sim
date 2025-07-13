"""
Agent archetypes for military simulation
Defines personality types, skill multipliers, and POI preferences for each archetype
"""

from typing import Dict

# Archetype definitions for different military personas
ARCHETYPES: Dict[str, Dict] = {
    "weapon_specialist": {
        "description": "Weapon enthusiast who loves training and armory work",
        "skill_multipliers": {
            "weapon_strength": 1.5,
            "management_skill": 0.8,
            "sociability": 0.9,
            "power": 1.4,
        },
        "poi_preferences": {
            "training": 2.0,
            "armory": 2.5,
            "food": 1.0,
            "rest": 0.8,
            "sleep": 1.0,
            "office": 0.5,
            "recreation": 0.7,
            "medical": 0.8,
            "fitness": 1.4,
            "library": 0.6,
            "workshop": 1.8,
            "communications": 0.6,
            "maintenance": 1.3,
            "outdoor": 1.9,
            "spiritual": 0.7,
            "logistics": 0.8,
        },
    },
    "natural_leader": {
        "description": "Born leader with management and strategy focus",
        "skill_multipliers": {
            "weapon_strength": 1.0,
            "management_skill": 1.6,
            "sociability": 1.3,
            "power": 1.1,
        },
        "poi_preferences": {
            "training": 1.2,
            "armory": 1.0,
            "food": 1.1,
            "rest": 0.9,
            "sleep": 1.0,
            "office": 2.0,
            "recreation": 1.0,
            "medical": 1.0,
            "fitness": 1.1,
            "library": 1.6,
            "workshop": 1.2,
            "communications": 1.8,
            "maintenance": 1.4,
            "outdoor": 1.3,
            "spiritual": 1.1,
            "logistics": 1.9,
        },
    },
    "social_butterfly": {
        "description": "Highly social person who excels in communication",
        "skill_multipliers": {
            "weapon_strength": 0.9,
            "management_skill": 1.2,
            "sociability": 1.7,
            "power": 0.8,
        },
        "poi_preferences": {
            "training": 1.0,
            "armory": 0.6,
            "food": 1.8,
            "rest": 1.0,
            "sleep": 1.0,
            "office": 1.3,
            "recreation": 2.2,
            "medical": 1.2,
            "fitness": 1.3,
            "library": 0.8,
            "workshop": 0.9,
            "communications": 2.1,
            "maintenance": 1.0,
            "outdoor": 1.4,
            "spiritual": 1.5,
            "logistics": 1.2,
        },
    },
    "scholar": {
        "description": "Intellectual type who prefers study and strategic planning",
        "skill_multipliers": {
            "weapon_strength": 0.8,
            "management_skill": 1.4,
            "sociability": 1.0,
            "power": 0.7,
        },
        "poi_preferences": {
            "training": 0.8,
            "armory": 0.7,
            "food": 1.0,
            "rest": 1.2,
            "sleep": 1.1,
            "office": 2.2,
            "recreation": 0.9,
            "medical": 1.0,
            "fitness": 0.7,
            "library": 2.5,
            "workshop": 1.1,
            "communications": 1.7,
            "maintenance": 0.8,
            "outdoor": 0.6,
            "spiritual": 1.3,
            "logistics": 1.6,
        },
    },
    "fitness_enthusiast": {
        "description": "Physical fitness focused with high energy",
        "skill_multipliers": {
            "weapon_strength": 1.3,
            "management_skill": 0.9,
            "sociability": 1.1,
            "power": 1.6,
        },
        "poi_preferences": {
            "training": 2.3,
            "armory": 1.1,
            "food": 1.2,
            "rest": 0.6,
            "sleep": 0.8,
            "office": 0.4,
            "recreation": 1.5,
            "medical": 0.9,
            "fitness": 2.5,
            "library": 0.5,
            "workshop": 1.2,
            "communications": 0.7,
            "maintenance": 1.1,
            "outdoor": 2.4,
            "spiritual": 0.8,
            "logistics": 0.6,
        },
    },
    "introvert": {
        "description": "Quiet type who prefers individual work and rest",
        "skill_multipliers": {
            "weapon_strength": 1.1,
            "management_skill": 1.0,
            "sociability": 0.7,
            "power": 1.0,
        },
        "poi_preferences": {
            "training": 1.0,
            "armory": 1.3,
            "food": 0.7,
            "rest": 2.0,
            "sleep": 1.8,
            "office": 1.4,
            "recreation": 0.5,
            "medical": 1.1,
            "fitness": 0.8,
            "library": 1.9,
            "workshop": 1.6,
            "communications": 0.8,
            "maintenance": 1.5,
            "outdoor": 0.7,
            "spiritual": 1.7,
            "logistics": 1.2,
        },
    },
}


def get_archetype_data(archetype: str) -> Dict:
    """Get data for a specific archetype
    
    Args:
        archetype: The archetype name
        
    Returns:
        Dictionary with archetype data or empty dict if not found
    """
    return ARCHETYPES.get(archetype, {})


def get_skill_multiplier(archetype: str, skill: str) -> float:
    """Get skill multiplier for a specific archetype and skill
    
    Args:
        archetype: The archetype name
        skill: The skill name
        
    Returns:
        Skill multiplier (defaults to 1.0 if not found)
    """
    archetype_data = get_archetype_data(archetype)
    return archetype_data.get("skill_multipliers", {}).get(skill, 1.0)


def get_poi_preference(archetype: str, poi_category: str) -> float:
    """Get POI preference for a specific archetype and POI category
    
    Args:
        archetype: The archetype name
        poi_category: The POI category
        
    Returns:
        POI preference multiplier (defaults to 1.0 if not found)
    """
    archetype_data = get_archetype_data(archetype)
    return archetype_data.get("poi_preferences", {}).get(poi_category, 1.0)