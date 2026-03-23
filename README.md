# 🏎️ AI-Driven Interactive Car Game (Computer Vision & Bezier Modeling)

This repository contains a 2D game project that replaces traditional keyboard inputs with **MediaPipe** and **OpenCV** based head-tracking for real-time control.

## 🛠️ Engineering & Technical Depth
* **Dynamic Motion Control:** Implemented a system where **Yaw** angles control steering and **Pitch** angles control acceleration/deceleration.
* **Signal Filtering:** Developed a `PitchActionBuffer` class to filter raw sensor data, preventing jitter and ensuring smooth speed transitions.
* **Mathematical Modeling (Bezier Curves):** Utilized quadratic and cubic **Bezier curves** to architect a high-resolution 3D face model within `bezier curve.py`.
* **Pose Estimation:** Integrated the **solvePnP** algorithm to map 2D camera coordinates to a 3D coordinate system for accurate head-turn ratios.
* **Game Engine Optimization:** Features pixel-perfect collision detection and a harmonic wave-based enemy spawning system to maintain consistent challenge levels.

## 🧠 Key Challenges Solved
1. **Latency Reduction:** Optimized the image processing pipeline to achieve low-latency input for real-time gameplay.
2. **Data Integrity:** Managed asynchronous data flows between the camera feed and the Pygame event loop.
3. **Modular Design:** Separated the vision-based movement logic (`get_movement.py`) from the core game engine for better maintainability.

## 🚀 Technologies Used
* **Python:** Core development language.
* **OpenCV & MediaPipe:** Face mesh tracking and pose estimation.
* **Pygame:** Game physics, rendering, and asset management.
* **NumPy:** Matrix calculations and 3D coordinate transformations.

---
## 🤝 Contributors
* **Enes Malik Dincer** - Core Game Engine, Physics, and Pitch/Yaw Logic.
* **[Hasan Serdal Köksal]** - Development of the Face Sensor Module and Bezier Curve Implementation.

*Special thanks to my colleague for their significant contribution to the computer vision integration of this project.*
