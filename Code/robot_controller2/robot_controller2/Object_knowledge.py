rectangle_polygon = {
    "frame": {
        "name": "plane_frame_0"
    },

    "nodes": [
    #Points
    {"id": "pt_0", "type": "corner", "data": ["pt_0_x", "pt_0_y", "pt_0_z"]},
    {"id": "pt_1", "type": "corner", "data": ["pt_1_x", "pt_1_y", "pt_1_z"]},
    {"id": "pt_2", "type": "corner", "data": ["pt_2_x", "pt_2_y", "pt_2_z"]},
    {"id": "pt_3", "type": "corner", "data": ["pt_3_x", "pt_3_y", "pt_3_z"]},

    #Segments
    {"id": "line_segment_0", "type": "segment", "length": "line_segment_0_length",
     "slope_angle_deg": "line_segment_0_slope", "corners": ["pt_0", "pt_1"], "edge_unit_vector": "line_segment_0_unit_vector"},
    {"id": "line_segment_1", "type": "segment", "length": "line_segment_1_length",
     "slope_angle_deg": "line_segment_1_slope", "corners": ["pt_1", "pt_2"], "edge_unit_vector": "line_segment_1_unit_vector"},
    {"id": "line_segment_2", "type": "segment", "length": "line_segment_2_length",
     "slope_angle_deg": "line_segment_2_slope", "corners": ["pt_2", "pt_3"], "edge_unit_vector": "line_segment_2_unit_vector"},
    {"id": "line_segment_3", "type": "segment", "length": "line_segment_3_length",
     "slope_angle_deg": "line_segment_3_slope", "corners": ["pt_3", "pt_0"], "edge_unit_vector": "line_segment_3_unit_vector"},

    {"id": f"line_segment_0_unit_vector",
     "type": "vector3",
     "data": [f"line_segment_0_unit_vector_x", f"line_segment_0_unit_vector_y", f"line_segment_0_unit_vector_z"]},
    {"id": f"line_segment_1_unit_vector",
    "type": "vector3",
    "data": [f"line_segment_1_unit_vector_x", f"line_segment_1_unit_vector_y", f"line_segment_1_unit_vector_z"]},
    {"id": f"line_segment_2_unit_vector",
    "type": "vector3",
    "data": [f"line_segment_2_unit_vector_x", f"line_segment_2_unit_vector_y", f"line_segment_2_unit_vector_z"]},
    {"id": f"line_segment_3_unit_vector",
    "type": "vector3",
    "data": [f"line_segment_3_unit_vector_x", f"line_segment_3_unit_vector_y", f"line_segment_3_unit_vector_z"]},


    #Polygon
    {
        "id": "polygon_0",
        "type": "polygon",
        "corners": ["pt_0", "pt_1", "pt_2", "pt_3"]
    },

    #Corner
    # angles(Assumption: angles are considered accross the plane / polygon and not using specific conventions(clockwise / counterclockwise))
    {
        "id": "corner_angle_0",
        "type": "2d_corner_angle",
        "angle": "corner_angle_0_deg",
        "polygon_id": "polygon_0",
        "corner": "pt_0",
        "is_refexive": "is_refexive_ang_0"
    },

    {
        "id": "corner_angle_1",
        "type": "2d_corner_angle",
        "angle": "corner_angle_1_deg",
        "polygon_id": "polygon_0",
        "corner": "pt_1",
        "is_refexive": "is_refexive_ang_1"
    },

    {
        "id": "corner_angle_2",
        "type": "2d_corner_angle",
        "angle": "corner_angle_2_deg",
        "polygon_id": "polygon_0",
        "corner": "pt_2",
        "is_refexive": "is_refexive_ang_2"
    },

    {
        "id": "corner_angle_3",
        "type": "2d_corner_angle",
        "angle": "corner_angle_3_deg",
        "polygon_id": "polygon_0",
        "corner": "pt_3",
        "is_refexive": "is_refexive_ang_3"
    },

    # Edge angle across polygons(assumption: measured between normals of two polygons)
    {
        "id": "dihedral_angle_0",
        "type": "2d_dihedral_angle",
        "angle": "dihedral_angle_0_deg",
        "segment": "line_segment_0",
        "polygons": ["polygon_0", "polygon_1"]
    },

    {
        "id": "dihedral_angle_1",
        "type": "2d_dihedral_angle",
        "angle": "dihedral_angle_1_deg",
        "segment": "line_segment_1",
        "polygons": ["polygon_0", "polygon_2"]
    },

    {
        "id": "dihedral_angle_2",
        "type": "2d_dihedral_angle",
        "angle": "dihedral_angle_2_deg",
        "segment": "line_segment_2",
        "polygons": ["polygon_0", "polygon_3"]
    },

    {
        "id": "dihedral_angle_3",
        "type": "2d_dihedral_angle",
        "angle": "dihedral_angle_3_deg",
        "segment": "line_segment_3",
        "polygons": ["polygon_0", "polygon_4"]
    }
    ],

    "data_structure": [
        {"id": "pt_0_x", "type": "float", "value": None},
        {"id": "pt_0_y", "type": "float", "value": None},
        {"id": "pt_0_z", "type": "float", "value": None},
        {"id": "pt_1_x", "type": "float", "value": None},
        {"id": "pt_1_y", "type": "float", "value": None},
        {"id": "pt_1_z", "type": "float", "value": None},
        {"id": "pt_2_x", "type": "float", "value": None},
        {"id": "pt_2_y", "type": "float", "value": None},
        {"id": "pt_2_z", "type": "float", "value": None},
        {"id": "pt_3_x", "type": "float", "value": None},
        {"id": "pt_3_y", "type": "float", "value": None},
        {"id": "pt_3_z", "type": "float", "value": None},
        {"id": "line_segment_0_length", "type": "float", "value": 0.34}, #0.6
        {"id": "line_segment_0_slope", "type": "float", "value": None},
        {"id": "line_segment_0_unit_vector_x", "type": "float", "value": None},
        {"id": "line_segment_0_unit_vector_y", "type": "float", "value": None},
        {"id": "line_segment_0_unit_vector_z", "type": "float", "value": None},
        {"id": "line_segment_1_length", "type": "float", "value": 0.6}, #0.34
        {"id": "line_segment_1_slope", "type": "float", "value": None},
        {"id": "line_segment_1_unit_vector_x", "type": "float", "value": None},
        {"id": "line_segment_1_unit_vector_y", "type": "float", "value": None},
        {"id": "line_segment_1_unit_vector_z", "type": "float", "value": None},
        {"id": "line_segment_2_length", "type": "float", "value": 0.34},
        {"id": "line_segment_2_slope", "type": "float", "value": None},
        {"id": "line_segment_2_unit_vector_x", "type": "float", "value": None},
        {"id": "line_segment_2_unit_vector_y", "type": "float", "value": None},
        {"id": "line_segment_2_unit_vector_z", "type": "float", "value": None},
        {"id": "line_segment_3_length", "type": "float", "value": 0.6},
        {"id": "line_segment_3_slope", "type": "float", "value": None},
        {"id": "line_segment_3_unit_vector_x", "type": "float", "value": None},
        {"id": "line_segment_3_unit_vector_y", "type": "float", "value": None},
        {"id": "line_segment_3_unit_vector_z", "type": "float", "value": None},
        {"id": "corner_angle_0_deg", "type": "float", "value": 90.0},
        {"id": "corner_angle_1_deg", "type": "float", "value": 90.0},
        {"id": "corner_angle_2_deg", "type": "float", "value": 90.0},
        {"id": "corner_angle_3_deg", "type": "float", "value": 90.0},
        {"id": "dihedral_angle_0_deg", "type": "float", "value": 270.0},
        {"id": "dihedral_angle_1_deg", "type": "float", "value": 90.0},
        {"id": "dihedral_angle_2_deg", "type": "float", "value": 90.0},
        {"id": "dihedral_angle_3_deg", "type": "float", "value": 90.0},
        {"id": "is_refexive_ang_0", "type": "bool", "value": False},
        {"id": "is_refexive_ang_1", "type": "bool", "value": False},
        {"id": "is_refexive_ang_2", "type": "bool", "value": False},
        {"id": "is_refexive_ang_3", "type": "bool", "value": False}
    ],

    "edges": [
    #Polygon edges
    ["polygon_0", "line_segment_0"],
    ["polygon_0", "line_segment_1"],
    ["polygon_0", "line_segment_2"],
    ["polygon_0", "line_segment_3"],

    #Polygon to corner connections
    ["polygon_0", "pt_0"],
    ["polygon_0", "pt_1"],
    ["polygon_0", "pt_2"],
    ["polygon_0", "pt_3"],

    #Local edges
    ["pt_0", "line_segment_0"],
    ["pt_1", "line_segment_0"],
    ["pt_1", "line_segment_1"],
    ["pt_2", "line_segment_1"],
    ["pt_2", "line_segment_2"],
    ["pt_3", "line_segment_2"],
    ["pt_3", "line_segment_3"],
    ["pt_0", "line_segment_3"],

    #Angle connections
    ["corner_angle_0", "pt_0"],
    ["corner_angle_1", "pt_1"],
    ["corner_angle_2", "pt_2"],
    ["corner_angle_3", "pt_3"],

    #Edge angle connections
    ["dihedral_angle_0", "line_segment_0"],
    ["dihedral_angle_1", "line_segment_1"],
    ["dihedral_angle_2", "line_segment_2"],
    ["dihedral_angle_3", "line_segment_3"]
    ]
}