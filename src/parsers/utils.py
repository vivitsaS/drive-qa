import json
from loguru import logger
from typing import Optional
main_data_path = '/Users/vivitsashankar/Desktop/workspace/drive-test/concatenated_data/concatenated_data.json'

# Mapping from serial number to scene token based on CSV order
SCENE_SERIAL_TO_TOKEN = {
    1: "cc8c0bf57f984915a77078b10eb33198",  # scene-0061
    2: "bebf5f5b2a674631ab5c88fd1aa9e87a",  # scene-0655
    3: "2fc3753772e241f2ab2cd16a784cc680",  # scene-0757
    4: "d25718445d89453381c659b9c8734939",  # scene-1077
    5: "de7d80a1f5fb4c3e82ce8a4f213b450a",  # scene-1094
    6: "e233467e827140efa4b42d2b4c435855"   # scene-1100
}

def fetch_scene_data(scene_identifier: str) -> dict:
    """
    Fetch scene data from the JSON file. 
    Identifier can be scene_token or scene serial number (1-6). 
    Serial number is the index of the scene in the CSV file order.
    
    Args:
        scene_identifier (str): Either a scene token or scene serial number (1-6)
        
    Returns:
        dict: Scene data if found, None otherwise
    """
    try:
        with open(main_data_path, 'r') as f:
            data = json.load(f)

        
        # Try to match by scene token first
        if scene_identifier in data:
            return data[scene_identifier]
        
        # Try to match by scene serial number
        try:
            serial_num = int(scene_identifier)
            if serial_num in SCENE_SERIAL_TO_TOKEN:
                scene_token = SCENE_SERIAL_TO_TOKEN[serial_num]
                if scene_token in data:
                    return data[scene_token]
        except ValueError:
            # scene_identifier is not a number, so it's not a valid serial number
            logger.error(f"Scene {scene_identifier} not found in data")
            pass
        
        logger.error(f"Scene {scene_identifier} not found in data")
        return None
        
    except FileNotFoundError:
        logger.error(f"Data file not found at {main_data_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {main_data_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        return None

def fetch_no_and_ids_of_keyframes(scene_identifier: str) -> dict:
    """
    Fetch the number of keyframes and the index of the keyframes for a given scene.
    """
    try:
        scene_data = fetch_scene_data(scene_identifier)
    except Exception as e:
        logger.error(f"Error fetching scene data: {e}")
        return None
    return {
        "no_of_keyframes": len(scene_data['key_frames']),
        "keyframes_ids": list(scene_data['key_frames'].keys())
    }
    
def fetch_scene_keyframe(scene_identifier: str, frame_identifier: str) -> dict:
    """
    Fetch scene keyframe data from the JSON file.
    The frame identifier will just be the number that is the index of the frame in the scene. So if there are 10 frames in the scene, the frame identifier will be 0 to 9. regardless of the token. so the keyframes are stored in the scene_data['key_frames'] dict. and will be present like- { {<token>}: {<frame_data>}, {<token>}: {<frame_data>}, ... }
    """
    scene_data = fetch_scene_data(scene_identifier)

    if scene_data is None:
        logger.error(f"Scene {scene_identifier} not found in data")
        return None
    # see the length of the key_frames dict
    key_frames_length = len(scene_data['key_frames'])
    if frame_identifier >= key_frames_length:
        logger.error(f"Frame {frame_identifier} index exceeds the length of the keyframes in scene {scene_identifier}. There are only {key_frames_length} keyframes in scene {scene_identifier}")
        return None
    else:
        all_keyframes = scene_data['key_frames']
        # Show only the keys (tokens) up to level 1
        keyframe_tokens = list(all_keyframes.keys())
        # now that we have the list of keyframe tokens, we can map it to the frame_identifier, which is just the number of the frame in the scene.
        keyframe_data = all_keyframes[keyframe_tokens[frame_identifier]]
        # logger.info(f"Keyframe data for frame {s}: {keyframe_data}")
        return keyframe_data

def fetch_scene_keyframe_qa_pairs(scene_identifier: str, frame_identifier: str, type: Optional[str] = None) -> dict:
    """
    Fetch scene keyframe QA pairs data from the JSON file.
    """
    # first get keyframe data
    keyframe_data = fetch_scene_keyframe(scene_identifier, frame_identifier)
    if keyframe_data is None:
        logger.error(f"Keyframe data not found for scene {scene_identifier} and frame {frame_identifier}")
        return None
    # now get the qa_pairs from the keyframe_data
    qa_pairs = keyframe_data['QA']
    if type is not None:
        qa_pairs = qa_pairs[type]
    return qa_pairs


def fetch_scene_keyframe_object_info(scene_token: str, frame_token: str) -> dict:
    """
    Fetch scene keyframe object info data from the JSON file.
    """
    return

def fetch_scene_keyframe_object_info_by_id(scene_token: str, frame_token: str, object_id: str) -> dict:
    """
    Fetch scene keyframe object info data from the JSON file by object id.
    """
    return

def fetch_scene_keyframe_object_info_by_category(scene_token: str, frame_token: str, category: str) -> dict:
    """
    Fetch scene keyframe object info data from the JSON file by category.
    """

