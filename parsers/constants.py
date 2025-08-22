"""
Constants for the parsers module.

Token mappings and other constants used by data loading and parsing.
"""

# Token mappings for scenes and keyframes
SCENE_TOKEN_MAPPINGS = {
    1: "cc8c0bf57f984915a77078b10eb33198",  # scene-0061
    2: "bebf5f5b2a674631ab5c88fd1aa9e87a",  # scene-0655
    3: "2fc3753772e241f2ab2cd16a784cc680",  # scene-0757
    4: "d25718445d89453381c659b9c8734939",  # scene-1077
    5: "de7d80a1f5fb4c3e82ce8a4f213b450a",  # scene-1094
    6: "e233467e827140efa4b42d2b4c435855"   # scene-1100
}

KEYFRAME_TOKEN_MAPPINGS = {
    "cc8c0bf57f984915a77078b10eb33198": {  # scene-0061
        1: "1e3d79dae62742a0ad64c91679863358",
        2: "378a3a3e9af346308ab9dff8ced46d9c",
        3: "4246e57f018745c9b2bc68feb3d71b58",
        4: "4711bcd34644420da8bc77163431888e",
        5: "88449a5cb1644a199c1c11f6ac034867",
        6: "aa581aac963a4fad848ac11fe66e8637",
        7: "b73152cab88f49d9ba195da81fde1809",
        8: "e0845f5322254dafadbbed75aaa07969"
    },
    "bebf5f5b2a674631ab5c88fd1aa9e87a": {  # scene-0655
        1: "23799328f65843fc9e0c71f2bfbe90ef",
        2: "5f3fc3c1d08b47cabc1bbca600abbfa8",
        3: "91d9058720eb4c25a172236e14e11085",
        4: "ed1eee39e3dd4c30a3d932e3ceaa92c2",
        5: "ff007cb7b78443e6887401d694f0d369"
    },
    "2fc3753772e241f2ab2cd16a784cc680": {  # scene-0757
        1: "00889f8a9549450aa2f32cf310a3e305",
        2: "0307afa1ec7e48ce90295acabf8cb6dd",
        3: "286532245ff646a9915070c8b402e487",
        4: "348c8122f47349429a6cd694dcac86e6",
        5: "8c0372d87281421aba897bec0084d0da",
        6: "a5addc41b07842768a2183abdd01cff8",
        7: "bad8a002637046178fe0f8216e5bc355",
        8: "e122a34ec57c4edf8311b40c3f85f863"
    },
    "d25718445d89453381c659b9c8734939": {  # scene-1077
        1: "8c8755ebb0bf4c519416323acad6cea1",
        2: "c0f7a7d51fbe46f9be8ca098da06b7fb",
        3: "e8b1863300964b2481fee312496d06d8",
        4: "ed2e7d24f2ad41abb3d37cd9a0ca8e89"
    },
    "de7d80a1f5fb4c3e82ce8a4f213b450a": {  # scene-1094
        1: "07175443e8444b8c8c4802fd004e33c1",
        2: "3b594831ff7f4e27a9889c7b7d0d8e92",
        3: "648fe1aeca944fb793b334cb6ee01854",
        4: "6cb024831cce4b6e8acf85afb7cece6e",
        5: "95c0a92d52584756832ec795d346e296",
        6: "98fa81992fb04e799c329eca220e7164",
        7: "f65ffdc408fb4a0c8ef0d1614b47dce8",
        8: "fe40762a54e1414da73de751877ad576"
    },
    "e233467e827140efa4b42d2b4c435855": {  # scene-1100
        1: "41550a1f0b0548e5823b14d95e807c7d",
        2: "61cfc5554a724824b0d34bddc147d890",
        3: "93f4f8ac1145418cadcf14dfd2a9cf78",
        4: "b902d29df24c4efda414a9a88ed57031",
        5: "d7cb9aa06de1442d8e2a22d562045cb4"
    }
} 