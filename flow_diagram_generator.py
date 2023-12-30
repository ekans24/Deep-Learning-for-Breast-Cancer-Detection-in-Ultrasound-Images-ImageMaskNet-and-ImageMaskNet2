import graphviz

def create_flow_diagram():
    dot = graphviz.Digraph('ImageMaskNet')

    # Define nodes and edges for the image branch
    dot.node('I0', 'Image Input\n[3-channel RGB]')
    dot.node('I1', 'Conv2d-32')
    dot.node('I2', 'ReLU')
    dot.node('I3', 'MaxPool2d')
    dot.node('I4', 'Conv2d-64')
    dot.node('I5', 'ReLU')
    dot.node('I6', 'MaxPool2d')
    dot.node('IF', 'Flatten')

    # Define the sequence of edges for the image branch
    for i in range(6):
        dot.edge(f'I{i}', f'I{i+1}')
    dot.edge('I6', 'IF')

    # Define nodes and edges for the mask branch
    dot.node('M0', 'Mask Input\n[1-channel Grayscale]')
    dot.node('M1', 'Conv2d-32')
    dot.node('M2', 'ReLU')
    dot.node('M3', 'MaxPool2d')
    dot.node('M4', 'Conv2d-64')
    dot.node('M5', 'ReLU')
    dot.node('M6', 'MaxPool2d')
    dot.node('MF', 'Flatten')

    # Define the sequence of edges for the mask branch
    for i in range(6):
        dot.edge(f'M{i}', f'M{i+1}')
    dot.edge('M6', 'MF')

    # Define nodes for concatenated features and the output
    dot.node('C', 'Concatenate\n[Flattened Features]')
    dot.node('FC1', 'Linear-128')
    dot.node('R', 'ReLU')
    dot.node('FC2', 'Linear-3')
    dot.node('O', 'Output\n[benign, malignant or normal]')

    # Define edges for concatenated features and the output
    dot.edge('IF', 'C')
    dot.edge('MF', 'C')
    dot.edges([('C', 'FC1'), ('FC1', 'R'), ('R', 'FC2'), ('FC2', 'O')])

    return dot

# Generate the diagram
diagram = create_flow_diagram()
diagram_path = 'ImageMaskNet_FlowDiagram'

# Add 'format='png'' to render the diagram in PNG format
diagram.render(diagram_path, format='png')

# Output the diagram path with the correct '.png' extension
print(diagram_path + '.png')
