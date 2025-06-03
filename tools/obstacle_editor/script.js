document.addEventListener('DOMContentLoaded', () => {
    // Canvas setup
    const canvas = document.getElementById('gridCanvas');
    const ctx = canvas.getContext('2d');
    let canvasWidth, canvasHeight;

    // Grid settings
    let gridSize = parseInt(document.getElementById('gridSize').value);
    let scaleFactor = 0.1; // 1 grid unit = 0.1 meters

    // Obstacles
    let obstacles = [];
    let selectedObstacleIndex = -1;

    // Interface elements
    const obstaclesList = document.getElementById('obstaclesList');
    const addObstacleBtn = document.getElementById('addObstacle');
    const removeObstacleBtn = document.getElementById('removeObstacle');
    const clearAllBtn = document.getElementById('clearAll');
    const exportJsonBtn = document.getElementById('exportJson');
    const radiusInput = document.getElementById('radius');
    const gridSizeInput = document.getElementById('gridSize');

    // Target position inputs
    const targetXInput = document.getElementById('targetX');
    const targetYInput = document.getElementById('targetY');
    const targetZInput = document.getElementById('targetZ');

    // Initialize the canvas
    initCanvas();

    // Event listeners
    window.addEventListener('resize', initCanvas);
    gridSizeInput.addEventListener('change', () => {
        gridSize = parseInt(gridSizeInput.value);
        drawGrid();
    });

    canvas.addEventListener('click', handleCanvasClick);
    addObstacleBtn.addEventListener('click', addObstacleFromUI);
    removeObstacleBtn.addEventListener('click', removeSelectedObstacle);
    clearAllBtn.addEventListener('click', clearAllObstacles);
    exportJsonBtn.addEventListener('click', exportToJson);

    // Functions
    function initCanvas() {
        const containerWidth = canvas.parentElement.clientWidth;
        const containerHeight = canvas.parentElement.clientHeight;

        canvas.width = containerWidth;
        canvas.height = containerHeight;

        canvasWidth = canvas.width;
        canvasHeight = canvas.height;

        drawGrid();
        drawObstacles();
    }

    function drawGrid() {
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);

        const cellSize = Math.min(canvasWidth, canvasHeight) / gridSize;
        const offsetX = (canvasWidth - cellSize * gridSize) / 2;
        const offsetY = (canvasHeight - cellSize * gridSize) / 2;

        // Draw grid lines
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;

        // Draw vertical lines
        for (let i = 0; i <= gridSize; i++) {
            const x = offsetX + i * cellSize;
            ctx.beginPath();
            ctx.moveTo(x, offsetY);
            ctx.lineTo(x, offsetY + cellSize * gridSize);
            ctx.stroke();
        }

        // Draw horizontal lines
        for (let i = 0; i <= gridSize; i++) {
            const y = offsetY + i * cellSize;
            ctx.beginPath();
            ctx.moveTo(offsetX, y);
            ctx.lineTo(offsetX + cellSize * gridSize, y);
            ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2;

        // X-axis
        const centerY = offsetY + cellSize * gridSize / 2;
        ctx.beginPath();
        ctx.moveTo(offsetX, centerY);
        ctx.lineTo(offsetX + cellSize * gridSize, centerY);
        ctx.stroke();

        // Y-axis
        const centerX = offsetX + cellSize * gridSize / 2;
        ctx.beginPath();
        ctx.moveTo(centerX, offsetY);
        ctx.lineTo(centerX, offsetY + cellSize * gridSize);
        ctx.stroke();

        // Draw grid markings
        drawGridMarkings(offsetX, offsetY, centerX, centerY, cellSize);

        // Draw target position
        const targetX = parseFloat(targetXInput.value);
        const targetY = parseFloat(targetYInput.value);
        const [canvasX, canvasY] = worldToCanvas(targetX, targetY);

        ctx.fillStyle = 'green';
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, 8, 0, Math.PI * 2);
        ctx.fill();

        drawObstacles();
    }

    function drawGridMarkings(offsetX, offsetY, centerX, centerY, cellSize) {
        const totalGridWidth = cellSize * gridSize;

        // Font setup for markings
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#444';

        // Calculate the step for grid markings (show every nth mark)
        const markStep = gridSize > 30 ? 5 : (gridSize > 20 ? 4 : 2);

        // X-axis markings
        for (let i = 0; i <= gridSize; i += markStep) {
            if (i === gridSize / 2) continue; // Skip center (0 point)

            const x = offsetX + i * cellSize;
            const worldX = ((i - gridSize / 2) * cellSize) / cellSize * scaleFactor;

            // Draw tick mark
            ctx.beginPath();
            ctx.moveTo(x, centerY - 5);
            ctx.lineTo(x, centerY + 5);
            ctx.stroke();

            // Draw coordinate value
            ctx.fillText(worldX.toFixed(1), x, centerY + 15);
        }

        // Y-axis markings
        for (let i = 0; i <= gridSize; i += markStep) {
            if (i === gridSize / 2) continue; // Skip center (0 point)

            const y = offsetY + i * cellSize;
            // Note Y is inverted in canvas coordinates
            const worldY = ((gridSize / 2 - i) * cellSize) / cellSize * scaleFactor;

            // Draw tick mark
            ctx.beginPath();
            ctx.moveTo(centerX - 5, y);
            ctx.lineTo(centerX + 5, y);
            ctx.stroke();

            // Draw coordinate value
            ctx.fillText(worldY.toFixed(1), centerX - 15, y);
        }

        // Mark the origin (0,0)
        ctx.fillStyle = '#000';
        ctx.fillText('0,0', centerX + 15, centerY + 15);
    }

    function drawObstacles() {
        obstacles.forEach((obstacle, index) => {
            const [canvasX, canvasY] = worldToCanvas(obstacle.x, obstacle.y);
            const radiusPixels = obstacle.radius / scaleFactor * (Math.min(canvasWidth, canvasHeight) / gridSize);

            ctx.fillStyle = index === selectedObstacleIndex ? 'rgba(255, 0, 0, 0.5)' : 'rgba(100, 100, 100, 0.7)';
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, radiusPixels, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = index === selectedObstacleIndex ? '#ff0000' : '#444';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    }

    function worldToCanvas(worldX, worldY) {
        const cellSize = Math.min(canvasWidth, canvasHeight) / gridSize;
        const offsetX = (canvasWidth - cellSize * gridSize) / 2;
        const offsetY = (canvasHeight - cellSize * gridSize) / 2;

        const centerX = offsetX + cellSize * gridSize / 2;
        const centerY = offsetY + cellSize * gridSize / 2;

        const canvasX = centerX + worldX / scaleFactor * cellSize;
        const canvasY = centerY - worldY / scaleFactor * cellSize; // Y is inverted in canvas

        return [canvasX, canvasY];
    }

    function canvasToWorld(canvasX, canvasY) {
        const cellSize = Math.min(canvasWidth, canvasHeight) / gridSize;
        const offsetX = (canvasWidth - cellSize * gridSize) / 2;
        const offsetY = (canvasHeight - cellSize * gridSize) / 2;

        const centerX = offsetX + cellSize * gridSize / 2;
        const centerY = offsetY + cellSize * gridSize / 2;

        const worldX = (canvasX - centerX) / cellSize * scaleFactor;
        const worldY = (centerY - canvasY) / cellSize * scaleFactor; // Y is inverted in canvas

        return [worldX, worldY];
    }

    function snapToGrid(worldX, worldY) {
        // Snap coordinates to the nearest grid point
        const snappedX = Math.round(worldX / scaleFactor) * scaleFactor;
        const snappedY = Math.round(worldY / scaleFactor) * scaleFactor;
        return [snappedX, snappedY];
    }

    function handleCanvasClick(event) {
        const rect = canvas.getBoundingClientRect();
        const canvasX = event.clientX - rect.left;
        const canvasY = event.clientY - rect.top;

        // Check if Shift key is pressed - if so, set target position instead of obstacle
        if (event.shiftKey) {
            const [rawWorldX, rawWorldY] = canvasToWorld(canvasX, canvasY);
            const [worldX, worldY] = snapToGrid(rawWorldX, rawWorldY);
            // Update target position inputs
            targetXInput.value = worldX.toFixed(2);
            targetYInput.value = worldY.toFixed(2);
            // Redraw with new target position
            drawGrid();
            return;
        }

        // Check if clicking on an existing obstacle (for selection)
        const clickedObstacleIndex = obstacles.findIndex(obstacle => {
            const [obsCanvasX, obsCanvasY] = worldToCanvas(obstacle.x, obstacle.y);
            const radiusPixels = obstacle.radius / scaleFactor * (Math.min(canvasWidth, canvasHeight) / gridSize);
            const distance = Math.sqrt((obsCanvasX - canvasX) ** 2 + (obsCanvasY - canvasY) ** 2);
            return distance <= radiusPixels;
        });

        if (clickedObstacleIndex !== -1) {
            selectedObstacleIndex = clickedObstacleIndex;
            updateObstaclesList();
            drawGrid();
            return;
        }

        // Add new obstacle at click position (snapped to grid)
        const [rawWorldX, rawWorldY] = canvasToWorld(canvasX, canvasY);
        const [worldX, worldY] = snapToGrid(rawWorldX, rawWorldY);
        const radius = parseFloat(radiusInput.value);
        const z = 0; // Default Z coordinate for 2D

        obstacles.push({
            x: worldX,
            y: worldY,
            z: z,
            radius: radius
        });

        selectedObstacleIndex = obstacles.length - 1;
        updateObstaclesList();
        drawGrid();
    }

    function addObstacleFromUI() {
        // Use the center of the grid as default position
        const worldX = 0;
        const worldY = 0;
        const z = 0; // Default Z coordinate for 2D
        const radius = parseFloat(radiusInput.value);

        obstacles.push({
            x: worldX,
            y: worldY,
            z: z,
            radius: radius
        });

        selectedObstacleIndex = obstacles.length - 1;
        updateObstaclesList();
        drawGrid();
    }

    function removeSelectedObstacle() {
        if (selectedObstacleIndex !== -1) {
            obstacles.splice(selectedObstacleIndex, 1);
            selectedObstacleIndex = -1;
            updateObstaclesList();
            drawGrid();
        }
    }

    function clearAllObstacles() {
        obstacles = [];
        selectedObstacleIndex = -1;
        updateObstaclesList();
        drawGrid();
    }

    function updateObstaclesList() {
        obstaclesList.innerHTML = '';

        obstacles.forEach((obstacle, index) => {
            const item = document.createElement('div');
            item.className = 'obstacle-item';
            if (index === selectedObstacleIndex) {
                item.classList.add('selected');
            }

            item.innerHTML = `
                <strong>Obstacle ${index + 1}</strong><br>
                X: ${obstacle.x.toFixed(2)}, Y: ${obstacle.y.toFixed(2)}, Z: ${obstacle.z.toFixed(2)}<br>
                Radius: ${obstacle.radius.toFixed(2)}
            `;

            item.addEventListener('click', () => {
                selectedObstacleIndex = index;
                updateObstaclesList();
                drawGrid();
            });

            obstaclesList.appendChild(item);
        });
    }

    function exportToJson() {
        // Format JSON according to the expected format in inverse_kinematics_2d.py
        const targetPosition = [parseFloat(targetXInput.value), parseFloat(targetYInput.value), parseFloat(targetZInput.value)];

        // Format obstacles to match the expected tensor format
        const obstacleData = obstacles.map(obs => [obs.x, obs.y, obs.z, obs.radius]);

        const jsonData = {
            target_position: targetPosition,
            quaternion: [0.0, 0.0, 0.0, 1.0], // Default quaternion for 2D
            obstacle_sphere: obstacleData
        };

        // Create a download link for the JSON file
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(jsonData, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "obstacles.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }

    // Initialize obstacles list
    updateObstaclesList();

    // Initial UI setup
    targetXInput.addEventListener('change', drawGrid);
    targetYInput.addEventListener('change', drawGrid);
    targetZInput.addEventListener('change', drawGrid);

    // Add a hint about setting target
    const gridContainer = document.querySelector('.grid-container');
    const hint = document.createElement('div');
    hint.className = 'hint';
    hint.textContent = 'Hold Shift + Click to set target position';
    gridContainer.appendChild(hint);
}); 