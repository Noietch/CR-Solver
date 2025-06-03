# 2D Obstacle Editor for DiffSoft

This is a web-based graphical interface for drawing 2D obstacles that can be directly used with the DiffSoft inverse kinematics solver.

## Features

- Grid-based interface for precise obstacle placement
- Interactive placement of circular obstacles
- Editable target position
- Export obstacles as JSON file compatible with inverse_kinematics_2d.py

## How to Use

1. Open `obstacle_editor.html` in a web browser
2. Configure the grid size and obstacle radius as needed
3. Click on the grid to place obstacles or use the "Add Obstacle" button
4. Set the target position using the input fields at the bottom
5. Click "Export JSON" to download the obstacles configuration

## Using the JSON with inverse_kinematics_2d.py

You can use the exported JSON file with the provided `load_obstacles.py` script:

```bash
python load_obstacles.py --json_file obstacles.json
```

This will:
1. Load the obstacles and target position from the JSON file
2. Set up the PCC model and IK solver
3. Run the inverse kinematics solver
4. Generate a visualization of the result

## Interface Controls

- **Grid Size**: Adjusts the resolution of the grid
- **Obstacle Radius**: Sets the default radius for new obstacles
- **Add Obstacle**: Adds a new obstacle at the center of the grid
- **Remove Selected**: Removes the currently selected obstacle
- **Clear All**: Removes all obstacles
- **Export JSON**: Exports the current configuration as a JSON file

## JSON Format

The exported JSON file has the following format:

```json
{
  "target_position": [x, y, z],
  "quaternion": [0.0, 0.0, 0.0, 1.0],
  "obstacle_sphere": [
    [x1, y1, z1, radius1],
    [x2, y2, z2, radius2],
    ...
  ]
}
```

This format is compatible with the `Goal` object used in the inverse_kinematics_2d.py script.

## File Structure

- `obstacle_editor.html`: Main HTML file for the web interface
- `style.css`: Styling for the interface
- `script.js`: JavaScript code for interface functionality
- `load_obstacles.py`: Python script to load and use the JSON file with the inverse kinematics solver 