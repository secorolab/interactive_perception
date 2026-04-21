import networkx as nx
import matplotlib.pyplot as plt

global pos

pos = None

def render_graph_visualization(rck_graph, center_node_id: str = 'polygon_0', k_value: float = 1.0, pause_time: float = 2):
    """
    Dynamically displays the current knowledge of polygon (rck) in an updating
    Matplotlib window, with a specified node centered in the plot and
    displays the individual values of each node.
    """
    global pos

    if center_node_id not in rck_graph.nodes:
        print(f"Error: The node '{center_node_id}' selected to be centered does not exist in the graph.")
        return

    # If pos is None or if the graph has new nodes not yet in pos, calculate an initial layout for all nodes.
    if pos is None or any(node not in pos for node in rck_graph.nodes):
        pos = nx.spring_layout(rck_graph, k=k_value, seed=42, iterations=50)

    # Explicitly set the center node's position to (0,0) in a temporary dictionary, This ensures its the anchor for the next layout calculation
    fixed_nodes = [center_node_id]
    initial_pos_for_fixed = {center_node_id: [0.0, 0.0]}

    # Merge existing pos with the fixed node's initial position. This preserves the positions of other existing nodes while ensuring the center node is at (0,0).
    combined_pos = {**pos, **initial_pos_for_fixed}

    # Recalculate layout, fixing the center node at its new (0,0) position.
    # Other nodes will adjust around it.
    pos = nx.spring_layout(rck_graph, pos=combined_pos, fixed=fixed_nodes, k=k_value, seed=42, iterations=5000)

    # Clear the current figure
    plt.clf()

    # Define node colors and labels based on node type and node values
    node_colors = []
    node_labels = {}
    for node_id, attributes in rck_graph.nodes(data=True):
        node_type = attributes.get('type')
        label_text = node_id

        # Customize the label based on the nodes type and attributes
        if node_type == 'corner':
            node_colors.append('skyblue')
            if 'data' in attributes and isinstance(attributes['data'], list):
                coords = ', '.join(
                    f"{val:.1f}" if isinstance(val, (int, float)) else str(val) for val in attributes['data'])
                label_text += f"\nData: ({coords})"
            #label_text = attributes['data']
        elif node_type == 'segment':
            node_colors.append('lightgreen')
            if 'length' in attributes:
                label_text += f"\nLength: {attributes['length']}"
            if 'slope_angle_deg' in attributes:
                label_text += f"\nSlope: {attributes['slope_angle_deg']}°"
        elif node_type == 'polygon':
            node_colors.append('purple')
            # if 'corners' in attributes and isinstance(attributes['corners'], list):
            #     label_text += f"\nCorners: {', '.join(attributes['corners'])}"
        elif node_type == 'contact_point':
            node_colors.append('gold')
            if 'data' in attributes and isinstance(attributes['data'], list):
                coords = ', '.join(
                    f"{val:.1f}" if isinstance(val, (int, float)) else str(val) for val in attributes['data'])
                label_text += f"\nData: ({coords})"
        elif node_type in ['2d_corner_angle', '2d_dihedral_angle']:
            node_colors.append('salmon')
            if 'angle' in attributes:
                label_text += f"\nAngle: {attributes['angle']}°"
        else:
            node_colors.append('gray')

        node_labels[node_id] = label_text

    # Draw the graph
    nx.draw_networkx_nodes(rck_graph, pos, node_color=node_colors,
                           node_size=900)
    nx.draw_networkx_edges(rck_graph, pos)
    nx.draw_networkx_labels(rck_graph, pos, labels=node_labels, font_size=8)

    plt.title("Robot Knowledge Graph")
    # Hide the axes
    plt.axis('off')

    # Redraw
    plt.draw()
    plt.pause(pause_time)


def create_graph_from_json(data):
    """
    Creates a NetworkX graph from a given JSON dictionary representing
    a polygon structure.
    """
    # Initialize an empty graph
    graph = nx.Graph()


    # This maps the string IDs to their actual values
    data_values = {item['id']: item['value'] for item in data['data_structure']}

    # Add nodes to the graph with their attributes
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']

        # Start with a base set of attributes
        attributes = {'type': node_type}

        # Add frame name if it the first node and relevant
        if 'frame' in data and data['frame']['name']:
            attributes['frame_name'] = data['frame']['name']

        # Add other attributes based on the node type
        if node_type == 'corner':
            # Replace placeholder IDs with actual values from data_values
            attributes['data'] = [data_values[d] for d in node['data']]
        elif node_type == 'segment':
            attributes['length'] = data_values.get(node['length'])
            attributes['slope_angle_deg'] = data_values.get(node['slope_angle_deg'])
            attributes['corners'] = node['corners']
        elif node_type == 'polygon':
            attributes['corners'] = node['corners']
        elif node_type == '2d_corner_angle':
            attributes['angle'] = data_values.get(node['angle'])
            attributes['polygon_id'] = node['polygon_id']
            attributes['corner'] = node['corner']
        elif node_type == '2d_dihedral_angle':
            attributes['angle'] = data_values.get(node['angle'])
            attributes['segment'] = node['segment']
            attributes['polygons'] = node['polygons']

        # Add the node to the graph with the collected attributes
        graph.add_node(node_id, **attributes)

    # The edges list provides direct connections between node IDs
    for edge in data['edges']:
        # Ensure the edge has two valid nodes
        if len(edge) == 2:
            graph.add_edge(edge[0], edge[1])

    return graph



def create_simplified_graph_from_json(data):
    """
    Creates a NetworkX graph containing only the polygon node from a given JSON dictionary.
    """
    graph = nx.Graph()

    # Find the polygon node and add it to the graph
    for node_data in data.get('nodes', []):
        if node_data.get('type') == 'polygon':
            node_id = node_data['id']
            # Get all attributes from the node data
            attributes = {key: value for key, value in node_data.items()}
            # Remove the id key as its not a node attribute
            attributes.pop('id', None)
            graph.add_node(node_id, **attributes)
            break

    return graph


def get_next_contact_point_id(graph):
    """Return the next available contact_point ID for the given segment in the belief_state."""
    existing_cps = [
        n for n in graph
        if graph.nodes[n].get("type") == "contact_point"
    ]

    indices = []
    for cp in existing_cps:
        try:
            indices.append(int(cp.split("_")[-1]))
        except ValueError:
            pass

    next_index = max(indices) + 1 if indices else 0
    return f"contact_point_{next_index}"

def get_next_2d_dihedral_angle_id(graph):
    """Return the next available dihedral_angle ID for the given segment."""
    existing_cps = [
        n for n in graph
        if graph.nodes[n].get("type") == "2d_dihedral_angle"
    ]

    indices = []
    for cp in existing_cps:
        try:
            indices.append(int(cp.split("_")[-1]))
        except ValueError:
            pass

    next_index = max(indices) + 1 if indices else 0
    return f"dihedral_angle_{next_index}"

def get_next_2d_corner_angle_id(graph):
    """Return the next available corner_angle ID for the given segment."""
    existing_cps = [
        n for n in graph
        if graph.nodes[n].get("type") == "2d_corner_angle"
    ]

    indices = []
    for cp in existing_cps:
        try:
            indices.append(int(cp.split("_")[-1]))
        except ValueError:
            pass

    next_index = max(indices) + 1 if indices else 0
    return f"corner_angle_{next_index}"

def get_next_line_segment_id(graph):
    """Return the next available line_segment ID for the given segment."""
    existing_cps = [
        n for n in graph
        if graph.nodes[n].get("type") == "segment"
    ]

    indices = []
    for cp in existing_cps:
        try:
            indices.append(int(cp.split("_")[-1]))
        except ValueError:
            pass

    next_index = max(indices) + 1 if indices else 0
    return f"line_segment_{next_index}"

def get_next_corner_id(graph):
    """Return the next available corner ID for the given segment."""
    existing_cps = [
        n for n in graph
        if graph.nodes[n].get("type") == "corner"
    ]

    indices = []
    for cp in existing_cps:
        try:
            indices.append(int(cp.split("_")[-1]))
        except ValueError:
            pass

    next_index = max(indices) + 1 if indices else 0
    return f"pt_{next_index}"

def get_last_corner_id(graph):
    count = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "corner")
    return f"pt_{count-1}" if count > 0 else None

def get_last_segment_id(graph):
    count = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "segment")
    return f"line_segment_{count}" if count > 0 else None


def get_two_contact_points_data(graph, segment_id):
    contact_points = [
        n for n in graph.neighbors(segment_id)
        if graph.nodes[n].get("type") == "contact_point"
    ][:2]

    # Extract only the data
    data_lists = [graph.nodes[cp].get("data") for cp in contact_points]

    # If less than 2 points find corner neighbors with data instead
    print("corner_data", get_all_corner_data(graph))
    if len(data_lists) < 2:
        corner_neighbors = [
            nbr for nbr in graph.neighbors(segment_id)
            if graph.nodes[nbr].get("type") == "corner"
        ]
        print("corner_neighbors: ", corner_neighbors)

        if len(corner_neighbors) >= 2:
            #print(graph.nodes[corner_neighbors].get("data"))
            data_lists.append(graph.nodes[corner_neighbors[-2]].get("data"))
            print(data_lists[1], data_lists[0])
            return data_lists[1], data_lists[0]


    while len(data_lists) < 2:
        data_lists.append(None)

    return data_lists[0], data_lists[1]


def get_two_contact_points_none(graph, segment_id):

    contact_points = [
        n for n in graph.neighbors(segment_id)
        if graph.nodes[n].get("type") == "contact_point"
    ][:2]

    # Extract only the data
    data_lists = [graph.nodes[cp].get("data") for cp in contact_points]

    while len(data_lists) < 2:
        data_lists.append(None)

    return data_lists[0], data_lists[1]


def get_one_contact_point_data(graph, segment_id):
    contact_point = [
        n for n in graph.neighbors(segment_id)
        if graph.nodes[n].get("type") == "contact_point"
    ][:1]

    # Extract only the data
    data_lists = [graph.nodes[cp].get("data") for cp in contact_point]


    print("data_lists: ", data_lists)
    return data_lists[0]

def add_new_corner(belief_state, object_knowledge, seg_id):
    corner_id = get_next_corner_id(belief_state)

    belief_state.add_node(corner_id, type="corner")
    belief_state.add_edge(corner_id, seg_id)
    belief_state.add_edge(corner_id, "polygon_0")

    # Automatically add a 2d_corner_angle as all corners should have one
    angle_node_id = get_next_2d_corner_angle_id(belief_state)
    belief_state.add_node(angle_node_id, type="2d_corner_angle")
    belief_state.add_edge(corner_id, angle_node_id)


    # If all angle information is the same, add angle information immediately
    angles = []
    for node_id, node_data in object_knowledge.nodes.items():
        if node_data.get("type") == "2d_corner_angle" and "angle" in node_data:
            angles.append(node_data["angle"])

    if angles and all(a == angles[0] for a in angles):
        belief_state.nodes[angle_node_id]["angle"] = angles[0]

    return corner_id

def add_new_line_segment(graph):
    segment_id = get_next_line_segment_id(graph)
    last_corner_id = get_last_corner_id(graph)

    graph.add_node(segment_id, type="segment")
    graph.add_edge(segment_id, "polygon_0")
    if last_corner_id:
        graph.add_edge(segment_id, last_corner_id)


    if get_last_corner_id(graph):
        graph.add_edge(segment_id, get_last_corner_id(graph))

    return segment_id

def add_new_contact_point(graph, data, line_segment):
    contact_point_id = get_next_contact_point_id(graph)

    graph.add_node(contact_point_id, type="contact_point")
    graph.add_edge(contact_point_id, line_segment)
    graph.nodes[contact_point_id]["data"] = data


    return

def get_all_segments(belief_state):
    segments = [
        seg_id for seg_id, data in belief_state.nodes(data=True)
        if data.get("type") == "segment"
    ]

    return segments

def get_all_corners(graph):
    corners = [
        c_id for c_id, data in graph.nodes(data=True)
        if data.get("type") == "corner"
    ]

    return corners


def get_all_corner_data(graph):
    corner_ids = get_all_corners(graph)
    return [graph.nodes[cid]["data"] for cid in corner_ids
            if cid in graph.nodes and "data" in graph.nodes[cid]]


def count_contact_points(graph, line_segment_id):
    count = 0
    for neighbor_node in graph.neighbors(line_segment_id):
        if graph.nodes[neighbor_node].get('type') == 'contact_point':
            count += 1

    return count
