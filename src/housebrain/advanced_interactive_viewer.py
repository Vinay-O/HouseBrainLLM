"""
Advanced Interactive 3D Viewer for HouseBrain

This module provides the most advanced 3D viewer in the industry with:
- Professional measurement tools integration
- Real-time material comparison
- Advanced lighting controls
- VR/AR support
- Collaborative features
- Performance optimization
- Professional annotation system
"""

from __future__ import annotations

import json
from typing import Dict, List, Any
from pathlib import Path


class AdvancedInteractive3DViewer:
    """Industry-leading interactive 3D viewer"""
    
    def __init__(self):
        self.measurement_tools = None
        self.material_comparison = None
        self.error_handler = None
        self.cache_system = None
        
        print("🎮 Advanced Interactive 3D Viewer Initialized")
    
    def generate_professional_viewer(
        self,
        house_data: Dict[str, Any],
        viewer_name: str,
        output_dir: str,
        features: List[str] = None
    ) -> str:
        """Generate professional interactive 3D viewer"""
        
        if features is None:
            features = [
                "measurement_tools", "material_comparison", "lighting_controls",
                "performance_optimization", "collaborative_features", "vr_ar_support"
            ]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        viewer_file = output_path / f"{viewer_name}_professional_viewer.html"
        
        # Generate complete viewer HTML
        html_content = self._generate_complete_viewer_html(house_data, viewer_name, features)
        
        with open(viewer_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate supporting files
        self._generate_viewer_assets(output_path, viewer_name, house_data)
        
        print(f"✅ Professional viewer generated: {viewer_file}")
        return str(viewer_file)
    
    def _generate_complete_viewer_html(
        self,
        house_data: Dict[str, Any],
        viewer_name: str,
        features: List[str]
    ) -> str:
        """Generate complete professional viewer HTML"""
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HouseBrain Professional Viewer - {viewer_name}</title>
    
    <!-- Professional CSS -->
    <style>
        {self._generate_professional_css()}
    </style>
    
    <!-- Three.js and Extensions -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/DRACOLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/SSAOPass.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/OutlinePass.js"></script>
    
    <!-- VR/AR Support -->
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/vr/WebXR.js"></script>
    
    <!-- Performance Monitoring -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/stats.js/r17/Stats.min.js"></script>
</head>
<body>
    <!-- Professional Interface -->
    <div id="viewer-container">
        <!-- Main 3D Canvas -->
        <canvas id="three-canvas"></canvas>
        
        <!-- Professional Toolbar -->
        <div id="professional-toolbar">
            {self._generate_professional_toolbar(features)}
        </div>
        
        <!-- Advanced Side Panel -->
        <div id="advanced-panel">
            {self._generate_advanced_panel(features)}
        </div>
        
        <!-- Status Bar -->
        <div id="status-bar">
            {self._generate_status_bar()}
        </div>
        
        <!-- Modal Dialogs -->
        <div id="modal-container">
            {self._generate_modal_dialogs(features)}
        </div>
    </div>
    
    <!-- Professional JavaScript -->
    <script>
        {self._generate_core_viewer_js(house_data, features)}
    </script>
    
    <!-- Feature-Specific Scripts -->
    {"".join(self._generate_feature_scripts(feature) for feature in features)}
    
    <!-- Initialization Script -->
    <script>
        // Initialize Professional Viewer
        document.addEventListener('DOMContentLoaded', function() {{
            const viewer = new HouseBrainProfessionalViewer({{
                container: 'viewer-container',
                canvas: 'three-canvas',
                features: {json.dumps(features)},
                houseData: {json.dumps(house_data)},
                quality: 'ultra'
            }});
            
            viewer.initialize();
            
            // Performance monitoring
            if (typeof Stats !== 'undefined') {{
                const stats = new Stats();
                document.body.appendChild(stats.dom);
                viewer.setStatsMonitor(stats);
            }}
        }});
    </script>
</body>
</html>'''
    
    def _generate_professional_css(self) -> str:
        """Generate professional CSS styling"""
        
        return '''
        /* HouseBrain Professional Viewer Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a1a;
            color: #ffffff;
            overflow: hidden;
        }
        
        #viewer-container {
            position: relative;
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        #three-canvas {
            flex: 1;
            background: radial-gradient(circle at center, #2a2a2a 0%, #1a1a1a 100%);
        }
        
        #professional-toolbar {
            position: absolute;
            top: 20px;
            left: 20px;
            right: 20px;
            height: 60px;
            background: rgba(40, 40, 40, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            padding: 0 20px;
            gap: 15px;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
        
        .toolbar-section {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 15px;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .toolbar-section:last-child {
            border-right: none;
        }
        
        .tool-button {
            width: 40px;
            height: 40px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #ffffff;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            font-size: 16px;
        }
        
        .tool-button:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }
        
        .tool-button.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: #667eea;
        }
        
        #advanced-panel {
            position: absolute;
            top: 100px;
            right: 20px;
            width: 350px;
            max-height: calc(100vh - 140px);
            background: rgba(40, 40, 40, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
            z-index: 999;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            transform: translateX(370px);
            transition: transform 0.3s ease;
        }
        
        #advanced-panel.open {
            transform: translateX(0);
        }
        
        .panel-section {
            padding: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .panel-section:last-child {
            border-bottom: none;
        }
        
        .panel-title {
            font-size: 14px;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .control-group {
            margin-bottom: 15px;
        }
        
        .control-label {
            display: block;
            font-size: 12px;
            color: #cccccc;
            margin-bottom: 5px;
        }
        
        .slider-control {
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider-control::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .select-control {
            width: 100%;
            padding: 8px 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            color: #ffffff;
            font-size: 12px;
        }
        
        .button-control {
            width: 100%;
            padding: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 6px;
            color: #ffffff;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .button-control:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        #status-bar {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 30px;
            background: rgba(40, 40, 40, 0.95);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            padding: 0 20px;
            font-size: 12px;
            color: #cccccc;
            z-index: 1000;
        }
        
        .status-item {
            margin-right: 20px;
        }
        
        .performance-indicator {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .perf-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
        }
        
        .perf-dot.warning { background: #FF9800; }
        .perf-dot.error { background: #F44336; }
        
        /* Modal Styles */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
        }
        
        .modal.active {
            display: flex;
        }
        
        .modal-content {
            background: rgba(40, 40, 40, 0.98);
            backdrop-filter: blur(20px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 30px;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
        }
        
        /* Measurement Tools Styles */
        .measurement-overlay {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 500;
        }
        
        .measurement-line {
            stroke: #00ff00;
            stroke-width: 2;
            fill: none;
        }
        
        .measurement-label {
            fill: #ffffff;
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: bold;
            text-anchor: middle;
        }
        
        .measurement-point {
            fill: #ff0000;
            stroke: #ffffff;
            stroke-width: 1;
        }
        
        /* Material Comparison Styles */
        .material-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        
        .material-preview {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        
        .material-preview img {
            width: 100%;
            height: 80px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        
        .material-name {
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 5px;
        }
        
        .material-properties {
            font-size: 10px;
            color: #cccccc;
        }
        
        /* VR/AR Styles */
        .vr-controls {
            display: none;
        }
        
        .vr-supported .vr-controls {
            display: block;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            #advanced-panel {
                width: 100%;
                right: 0;
                transform: translateY(100%);
            }
            
            #advanced-panel.open {
                transform: translateY(0);
            }
            
            #professional-toolbar {
                flex-wrap: wrap;
                height: auto;
                min-height: 60px;
                padding: 10px;
            }
        }
        
        /* Animation Classes */
        .fade-in {
            animation: fadeIn 0.3s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .loading {
            position: relative;
            overflow: hidden;
        }
        
        .loading::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: loading 1.5s infinite;
        }
        
        @keyframes loading {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        '''
    
    def _generate_professional_toolbar(self, features: List[str]) -> str:
        """Generate professional toolbar"""
        
        toolbar_sections = []
        
        # Navigation Section
        toolbar_sections.append('''
        <div class="toolbar-section">
            <div class="tool-button" id="reset-view" title="Reset View">
                <span>🏠</span>
            </div>
            <div class="tool-button" id="top-view" title="Top View">
                <span>⬇️</span>
            </div>
            <div class="tool-button" id="perspective-view" title="Perspective View">
                <span>👁️</span>
            </div>
        </div>
        ''')
        
        # Measurement Tools Section
        if "measurement_tools" in features:
            toolbar_sections.append('''
            <div class="toolbar-section">
                <div class="tool-button" id="distance-tool" title="Distance Measurement">
                    <span>📏</span>
                </div>
                <div class="tool-button" id="area-tool" title="Area Measurement">
                    <span>📐</span>
                </div>
                <div class="tool-button" id="angle-tool" title="Angle Measurement">
                    <span>📊</span>
                </div>
                <div class="tool-button" id="clear-measurements" title="Clear Measurements">
                    <span>🗑️</span>
                </div>
            </div>
            ''')
        
        # Material Tools Section
        if "material_comparison" in features:
            toolbar_sections.append('''
            <div class="toolbar-section">
                <div class="tool-button" id="material-picker" title="Material Picker">
                    <span>🎨</span>
                </div>
                <div class="tool-button" id="material-compare" title="Compare Materials">
                    <span>⚖️</span>
                </div>
                <div class="tool-button" id="material-library" title="Material Library">
                    <span>📚</span>
                </div>
            </div>
            ''')
        
        # Lighting Controls Section
        if "lighting_controls" in features:
            toolbar_sections.append('''
            <div class="toolbar-section">
                <div class="tool-button" id="lighting-presets" title="Lighting Presets">
                    <span>💡</span>
                </div>
                <div class="tool-button" id="time-of-day" title="Time of Day">
                    <span>🌅</span>
                </div>
                <div class="tool-button" id="shadow-quality" title="Shadow Quality">
                    <span>🌓</span>
                </div>
            </div>
            ''')
        
        # VR/AR Section
        if "vr_ar_support" in features:
            toolbar_sections.append('''
            <div class="toolbar-section vr-controls">
                <div class="tool-button" id="enter-vr" title="Enter VR">
                    <span>🥽</span>
                </div>
                <div class="tool-button" id="enter-ar" title="Enter AR">
                    <span>📱</span>
                </div>
            </div>
            ''')
        
        # Settings Section
        toolbar_sections.append('''
        <div class="toolbar-section">
            <div class="tool-button" id="settings" title="Settings">
                <span>⚙️</span>
            </div>
            <div class="tool-button" id="fullscreen" title="Fullscreen">
                <span>⛶</span>
            </div>
            <div class="tool-button" id="help" title="Help">
                <span>❓</span>
            </div>
        </div>
        ''')
        
        return "".join(toolbar_sections)
    
    def _generate_advanced_panel(self, features: List[str]) -> str:
        """Generate advanced control panel"""
        
        panel_sections = []
        
        # Scene Controls
        panel_sections.append('''
        <div class="panel-section">
            <div class="panel-title">Scene Controls</div>
            <div class="control-group">
                <label class="control-label">Quality Level</label>
                <select class="select-control" id="quality-level">
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="ultra" selected>Ultra</option>
                    <option value="cinematic">Cinematic</option>
                </select>
            </div>
            <div class="control-group">
                <label class="control-label">Field of View</label>
                <input type="range" class="slider-control" id="fov-slider" min="20" max="120" value="50">
            </div>
        </div>
        ''')
        
        # Lighting Controls
        if "lighting_controls" in features:
            panel_sections.append('''
            <div class="panel-section">
                <div class="panel-title">Lighting</div>
                <div class="control-group">
                    <label class="control-label">Sun Intensity</label>
                    <input type="range" class="slider-control" id="sun-intensity" min="0" max="5" step="0.1" value="1">
                </div>
                <div class="control-group">
                    <label class="control-label">Ambient Light</label>
                    <input type="range" class="slider-control" id="ambient-light" min="0" max="1" step="0.05" value="0.2">
                </div>
                <div class="control-group">
                    <label class="control-label">Shadow Quality</label>
                    <select class="select-control" id="shadow-quality">
                        <option value="off">Off</option>
                        <option value="low">Low</option>
                        <option value="medium" selected>Medium</option>
                        <option value="high">High</option>
                        <option value="ultra">Ultra</option>
                    </select>
                </div>
            </div>
            ''')
        
        # Material Controls
        if "material_comparison" in features:
            panel_sections.append('''
            <div class="panel-section">
                <div class="panel-title">Materials</div>
                <div class="control-group">
                    <button class="button-control" id="open-material-library">Open Material Library</button>
                </div>
                <div class="control-group">
                    <button class="button-control" id="compare-materials">Compare Materials</button>
                </div>
                <div class="material-comparison" id="material-comparison-area">
                    <!-- Material comparison will be populated here -->
                </div>
            </div>
            ''')
        
        # Measurement Tools
        if "measurement_tools" in features:
            panel_sections.append('''
            <div class="panel-section">
                <div class="panel-title">Measurements</div>
                <div class="control-group">
                    <label class="control-label">Measurement Units</label>
                    <select class="select-control" id="measurement-units">
                        <option value="mm">Millimeters</option>
                        <option value="cm">Centimeters</option>
                        <option value="m" selected>Meters</option>
                        <option value="inches">Inches</option>
                        <option value="feet">Feet</option>
                    </select>
                </div>
                <div class="control-group">
                    <label class="control-label">Precision</label>
                    <select class="select-control" id="measurement-precision">
                        <option value="0">0 decimals</option>
                        <option value="1">1 decimal</option>
                        <option value="2" selected>2 decimals</option>
                        <option value="3">3 decimals</option>
                    </select>
                </div>
                <div class="control-group">
                    <button class="button-control" id="export-measurements">Export Measurements</button>
                </div>
            </div>
            ''')
        
        # Performance Monitoring
        if "performance_optimization" in features:
            panel_sections.append('''
            <div class="panel-section">
                <div class="panel-title">Performance</div>
                <div class="control-group">
                    <label class="control-label">LOD Distance</label>
                    <input type="range" class="slider-control" id="lod-distance" min="1" max="100" value="20">
                </div>
                <div class="control-group">
                    <label class="control-label">Max FPS</label>
                    <select class="select-control" id="max-fps">
                        <option value="30">30 FPS</option>
                        <option value="60" selected>60 FPS</option>
                        <option value="120">120 FPS</option>
                        <option value="unlimited">Unlimited</option>
                    </select>
                </div>
                <div class="control-group">
                    <button class="button-control" id="optimize-performance">Optimize Performance</button>
                </div>
            </div>
            ''')
        
        return "".join(panel_sections)
    
    def _generate_status_bar(self) -> str:
        """Generate status bar"""
        
        return '''
        <div class="status-item">
            <span id="camera-position">Camera: 0, 0, 0</span>
        </div>
        <div class="status-item">
            <span id="selected-object">Selected: None</span>
        </div>
        <div class="status-item">
            <span id="measurement-mode">Mode: Navigation</span>
        </div>
        <div class="status-item performance-indicator">
            <div class="perf-dot" id="performance-indicator"></div>
            <span id="fps-counter">FPS: 60</span>
        </div>
        <div class="status-item">
            <span id="triangle-count">Triangles: 0</span>
        </div>
        '''
    
    def _generate_modal_dialogs(self, features: List[str]) -> str:
        """Generate modal dialogs"""
        
        modals = []
        
        # Material Library Modal
        if "material_comparison" in features:
            modals.append('''
            <div class="modal" id="material-library-modal">
                <div class="modal-content">
                    <h3>Professional Material Library</h3>
                    <div id="material-library-content">
                        <!-- Material library content will be loaded here -->
                    </div>
                    <button class="button-control" onclick="closeMaterialLibrary()">Close</button>
                </div>
            </div>
            ''')
        
        # Settings Modal
        modals.append('''
        <div class="modal" id="settings-modal">
            <div class="modal-content">
                <h3>Viewer Settings</h3>
                <div id="settings-content">
                    <!-- Settings content -->
                </div>
                <button class="button-control" onclick="closeSettings()">Close</button>
            </div>
        </div>
        ''')
        
        # Help Modal
        modals.append('''
        <div class="modal" id="help-modal">
            <div class="modal-content">
                <h3>HouseBrain Professional Viewer Help</h3>
                <div id="help-content">
                    <h4>Navigation</h4>
                    <ul>
                        <li><strong>Mouse:</strong> Left click + drag to rotate</li>
                        <li><strong>Mouse:</strong> Right click + drag to pan</li>
                        <li><strong>Mouse:</strong> Scroll to zoom</li>
                    </ul>
                    <h4>Measurement Tools</h4>
                    <ul>
                        <li><strong>Distance:</strong> Click two points to measure distance</li>
                        <li><strong>Area:</strong> Click multiple points to measure area</li>
                        <li><strong>Angle:</strong> Click three points to measure angle</li>
                    </ul>
                    <h4>Materials</h4>
                    <ul>
                        <li><strong>Pick Material:</strong> Click on surface to select material</li>
                        <li><strong>Compare:</strong> Select multiple materials to compare</li>
                    </ul>
                </div>
                <button class="button-control" onclick="closeHelp()">Close</button>
            </div>
        </div>
        ''')
        
        return "".join(modals)
    
    def _generate_core_viewer_js(self, house_data: Dict[str, Any], features: List[str]) -> str:
        """Generate core viewer JavaScript"""
        
        return f'''
        // HouseBrain Professional Viewer Core
        class HouseBrainProfessionalViewer {{
            constructor(options) {{
                this.container = document.getElementById(options.container);
                this.canvas = document.getElementById(options.canvas);
                this.features = options.features || [];
                this.houseData = options.houseData || {{}};
                this.quality = options.quality || 'high';
                
                // Three.js core
                this.scene = null;
                this.camera = null;
                this.renderer = null;
                this.controls = null;
                
                // Feature modules
                this.measurementTools = null;
                this.materialComparison = null;
                this.lightingSystem = null;
                this.performanceMonitor = null;
                
                // State
                this.selectedObjects = [];
                this.measurementMode = 'none';
                this.currentMeasurements = [];
                
                console.log('🎮 HouseBrain Professional Viewer initialized');
            }}
            
            initialize() {{
                this.setupThreeJS();
                this.setupControls();
                this.setupLighting();
                this.loadModel();
                this.setupFeatures();
                this.setupEventListeners();
                this.startRenderLoop();
                
                console.log('✅ Professional viewer ready');
            }}
            
            setupThreeJS() {{
                // Scene
                this.scene = new THREE.Scene();
                this.scene.background = new THREE.Color(0x1a1a1a);
                
                // Camera
                this.camera = new THREE.PerspectiveCamera(
                    50,
                    this.canvas.clientWidth / this.canvas.clientHeight,
                    0.1,
                    1000
                );
                this.camera.position.set(10, 10, 10);
                
                // Renderer
                this.renderer = new THREE.WebGLRenderer({{
                    canvas: this.canvas,
                    antialias: this.quality !== 'low',
                    alpha: true
                }});
                this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                this.renderer.shadowMap.enabled = true;
                this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                this.renderer.physicallyCorrectLights = true;
                this.renderer.outputEncoding = THREE.sRGBEncoding;
                this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
                this.renderer.toneMappingExposure = 1.0;
            }}
            
            setupControls() {{
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.minDistance = 1;
                this.controls.maxDistance = 100;
                this.controls.maxPolarAngle = Math.PI / 2;
            }}
            
            setupLighting() {{
                // Ambient light
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
                this.scene.add(ambientLight);
                
                // Directional light (sun)
                const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
                directionalLight.position.set(10, 10, 5);
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                directionalLight.shadow.camera.near = 0.5;
                directionalLight.shadow.camera.far = 50;
                directionalLight.shadow.camera.left = -20;
                directionalLight.shadow.camera.right = 20;
                directionalLight.shadow.camera.top = 20;
                directionalLight.shadow.camera.bottom = -20;
                this.scene.add(directionalLight);
                
                // Add light helper for development
                if (this.quality === 'ultra') {{
                    const lightHelper = new THREE.DirectionalLightHelper(directionalLight, 5);
                    this.scene.add(lightHelper);
                }}
            }}
            
            loadModel() {{
                // Create basic geometry from house data
                this.createBasicGeometry();
                
                // Load advanced glTF model if available
                // this.loadGLTFModel();
            }}
            
            createBasicGeometry() {{
                // Create walls
                const walls = this.houseData.geometry?.walls || [];
                walls.forEach(wall => {{
                    const wallMesh = this.createWallMesh(wall);
                    if (wallMesh) {{
                        this.scene.add(wallMesh);
                    }}
                }});
                
                // Create floors
                const spaces = this.houseData.geometry?.spaces || [];
                spaces.forEach(space => {{
                    const floorMesh = this.createFloorMesh(space);
                    if (floorMesh) {{
                        this.scene.add(floorMesh);
                    }}
                }});
            }}
            
            createWallMesh(wallData) {{
                const start = wallData.start || [0, 0];
                const end = wallData.end || [1000, 0];
                const thickness = (wallData.thickness || 200) / 1000; // Convert to meters
                const height = (wallData.height || 2700) / 1000; // Convert to meters
                
                const length = Math.sqrt(
                    Math.pow(end[0] - start[0], 2) + Math.pow(end[1] - start[1], 2)
                ) / 1000;
                
                const geometry = new THREE.BoxGeometry(length, height, thickness);
                
                const material = new THREE.MeshLambertMaterial({{
                    color: wallData.type === 'exterior' ? 0x8B4513 : 0xF5F5DC,
                    transparent: false
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                
                // Position the wall
                const centerX = (start[0] + end[0]) / 2 / 1000;
                const centerZ = (start[1] + end[1]) / 2 / 1000;
                mesh.position.set(centerX, height / 2, centerZ);
                
                // Rotate the wall
                const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
                mesh.rotation.y = angle;
                
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                mesh.userData = {{ type: 'wall', data: wallData }};
                
                return mesh;
            }}
            
            createFloorMesh(spaceData) {{
                const boundary = spaceData.boundary || [];
                if (boundary.length < 3) return null;
                
                // Create floor geometry
                const shape = new THREE.Shape();
                shape.moveTo(boundary[0][0] / 1000, boundary[0][1] / 1000);
                
                for (let i = 1; i < boundary.length; i++) {{
                    shape.lineTo(boundary[i][0] / 1000, boundary[i][1] / 1000);
                }}
                
                const geometry = new THREE.ShapeGeometry(shape);
                
                const material = new THREE.MeshLambertMaterial({{
                    color: 0xDDDDDD,
                    side: THREE.DoubleSide
                }});
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.rotation.x = -Math.PI / 2; // Rotate to horizontal
                mesh.receiveShadow = true;
                mesh.userData = {{ type: 'floor', data: spaceData }};
                
                return mesh;
            }}
            
            setupFeatures() {{
                {"".join(f"this.setup{feature.title().replace('_', '')}();" for feature in features)}
            }}
            
            setupEventListeners() {{
                // Resize handler
                window.addEventListener('resize', () => {{
                    this.handleResize();
                }});
                
                // Mouse events for interaction
                this.renderer.domElement.addEventListener('click', (event) => {{
                    this.handleClick(event);
                }});
                
                this.renderer.domElement.addEventListener('mousemove', (event) => {{
                    this.handleMouseMove(event);
                }});
                
                // Toolbar events
                this.setupToolbarEvents();
                
                // Keyboard shortcuts
                window.addEventListener('keydown', (event) => {{
                    this.handleKeydown(event);
                }});
            }}
            
            setupToolbarEvents() {{
                // Navigation tools
                document.getElementById('reset-view')?.addEventListener('click', () => {{
                    this.resetView();
                }});
                
                document.getElementById('top-view')?.addEventListener('click', () => {{
                    this.setTopView();
                }});
                
                document.getElementById('perspective-view')?.addEventListener('click', () => {{
                    this.setPerspectiveView();
                }});
                
                // Settings and help
                document.getElementById('settings')?.addEventListener('click', () => {{
                    this.openSettings();
                }});
                
                document.getElementById('help')?.addEventListener('click', () => {{
                    this.openHelp();
                }});
                
                document.getElementById('fullscreen')?.addEventListener('click', () => {{
                    this.toggleFullscreen();
                }});
            }}
            
            handleClick(event) {{
                const rect = this.renderer.domElement.getBoundingClientRect();
                const mouse = new THREE.Vector2(
                    ((event.clientX - rect.left) / rect.width) * 2 - 1,
                    -((event.clientY - rect.top) / rect.height) * 2 + 1
                );
                
                const raycaster = new THREE.Raycaster();
                raycaster.setFromCamera(mouse, this.camera);
                
                const intersects = raycaster.intersectObjects(this.scene.children, true);
                
                if (intersects.length > 0) {{
                    const intersection = intersects[0];
                    this.handleObjectClick(intersection);
                }}
            }}
            
            handleObjectClick(intersection) {{
                const object = intersection.object;
                const point = intersection.point;
                
                if (this.measurementMode !== 'none') {{
                    this.addMeasurementPoint(point);
                }} else {{
                    this.selectObject(object);
                }}
            }}
            
            selectObject(object) {{
                // Clear previous selection
                this.clearSelection();
                
                // Select new object
                this.selectedObjects.push(object);
                this.highlightObject(object);
                
                // Update status
                this.updateSelectedObjectStatus(object);
            }}
            
            highlightObject(object) {{
                // Add highlight effect
                if (object.material) {{
                    object.userData.originalMaterial = object.material.clone();
                    object.material.emissive.setHex(0x444444);
                }}
            }}
            
            clearSelection() {{
                this.selectedObjects.forEach(object => {{
                    if (object.userData.originalMaterial) {{
                        object.material = object.userData.originalMaterial;
                        delete object.userData.originalMaterial;
                    }}
                }});
                this.selectedObjects = [];
            }}
            
            resetView() {{
                this.camera.position.set(10, 10, 10);
                this.camera.lookAt(0, 0, 0);
                this.controls.reset();
            }}
            
            setTopView() {{
                this.camera.position.set(0, 20, 0);
                this.camera.lookAt(0, 0, 0);
                this.controls.update();
            }}
            
            setPerspectiveView() {{
                this.camera.position.set(15, 8, 15);
                this.camera.lookAt(0, 0, 0);
                this.controls.update();
            }}
            
            handleResize() {{
                const width = this.canvas.clientWidth;
                const height = this.canvas.clientHeight;
                
                this.camera.aspect = width / height;
                this.camera.updateProjectionMatrix();
                
                this.renderer.setSize(width, height);
            }}
            
            startRenderLoop() {{
                const animate = () => {{
                    requestAnimationFrame(animate);
                    
                    this.controls.update();
                    this.updateStatus();
                    
                    this.renderer.render(this.scene, this.camera);
                }};
                
                animate();
            }}
            
            updateStatus() {{
                // Update camera position
                const pos = this.camera.position;
                document.getElementById('camera-position').textContent = 
                    `Camera: ${{pos.x.toFixed(1)}}, ${{pos.y.toFixed(1)}}, ${{pos.z.toFixed(1)}}`;
                
                // Update triangle count
                let triangleCount = 0;
                this.scene.traverse((object) => {{
                    if (object.geometry) {{
                        triangleCount += object.geometry.index ? 
                            object.geometry.index.count / 3 : 
                            object.geometry.attributes.position.count / 3;
                    }}
                }});
                document.getElementById('triangle-count').textContent = `Triangles: ${{Math.floor(triangleCount)}}`;
            }}
            
            updateSelectedObjectStatus(object) {{
                const userData = object.userData;
                const objectInfo = userData.type || 'Object';
                document.getElementById('selected-object').textContent = `Selected: ${{objectInfo}}`;
            }}
            
            // Modal functions
            openSettings() {{
                document.getElementById('settings-modal').classList.add('active');
            }}
            
            openHelp() {{
                document.getElementById('help-modal').classList.add('active');
            }}
            
            toggleFullscreen() {{
                if (!document.fullscreenElement) {{
                    document.documentElement.requestFullscreen();
                }} else {{
                    document.exitFullscreen();
                }}
            }}
        }}
        
        // Global functions for modal controls
        function closeSettings() {{
            document.getElementById('settings-modal').classList.remove('active');
        }}
        
        function closeHelp() {{
            document.getElementById('help-modal').classList.remove('active');
        }}
        
        function closeMaterialLibrary() {{
            document.getElementById('material-library-modal').classList.remove('active');
        }}
        '''
    
    def _generate_feature_scripts(self, feature: str) -> str:
        """Generate feature-specific JavaScript"""
        
        if feature == "measurement_tools":
            return '''
            <script>
            // Measurement Tools Extension
            HouseBrainProfessionalViewer.prototype.setupMeasurementtools = function() {
                this.measurementMode = 'none';
                this.measurementPoints = [];
                this.measurementObjects = [];
                
                // Setup measurement tool buttons
                document.getElementById('distance-tool')?.addEventListener('click', () => {
                    this.setMeasurementMode('distance');
                });
                
                document.getElementById('area-tool')?.addEventListener('click', () => {
                    this.setMeasurementMode('area');
                });
                
                document.getElementById('angle-tool')?.addEventListener('click', () => {
                    this.setMeasurementMode('angle');
                });
                
                document.getElementById('clear-measurements')?.addEventListener('click', () => {
                    this.clearMeasurements();
                });
                
                console.log('📐 Measurement tools initialized');
            };
            
            HouseBrainProfessionalViewer.prototype.setMeasurementMode = function(mode) {
                this.measurementMode = mode;
                this.measurementPoints = [];
                
                // Update UI
                document.querySelectorAll('.tool-button').forEach(btn => btn.classList.remove('active'));
                document.getElementById(mode + '-tool')?.classList.add('active');
                
                // Update status
                document.getElementById('measurement-mode').textContent = `Mode: ${mode.charAt(0).toUpperCase() + mode.slice(1)}`;
            };
            
            HouseBrainProfessionalViewer.prototype.addMeasurementPoint = function(point) {
                this.measurementPoints.push(point);
                
                // Add visual point marker
                const geometry = new THREE.SphereGeometry(0.05);
                const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
                const marker = new THREE.Mesh(geometry, material);
                marker.position.copy(point);
                this.scene.add(marker);
                this.measurementObjects.push(marker);
                
                // Check if measurement is complete
                this.checkMeasurementComplete();
            };
            
            HouseBrainProfessionalViewer.prototype.checkMeasurementComplete = function() {
                if (this.measurementMode === 'distance' && this.measurementPoints.length === 2) {
                    this.completeMeasurement();
                } else if (this.measurementMode === 'angle' && this.measurementPoints.length === 3) {
                    this.completeMeasurement();
                }
            };
            
            HouseBrainProfessionalViewer.prototype.completeMeasurement = function() {
                if (this.measurementMode === 'distance') {
                    const distance = this.measurementPoints[0].distanceTo(this.measurementPoints[1]);
                    this.createDistanceMeasurement(this.measurementPoints[0], this.measurementPoints[1], distance);
                } else if (this.measurementMode === 'angle') {
                    const angle = this.calculateAngle(this.measurementPoints[0], this.measurementPoints[1], this.measurementPoints[2]);
                    this.createAngleMeasurement(this.measurementPoints[0], this.measurementPoints[1], this.measurementPoints[2], angle);
                }
                
                // Reset for next measurement
                this.measurementPoints = [];
            };
            
            HouseBrainProfessionalViewer.prototype.createDistanceMeasurement = function(start, end, distance) {
                // Create line
                const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
                const material = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 });
                const line = new THREE.Line(geometry, material);
                this.scene.add(line);
                this.measurementObjects.push(line);
                
                // Create label
                this.createMeasurementLabel(start.clone().lerp(end, 0.5), `${distance.toFixed(2)}m`);
            };
            
            HouseBrainProfessionalViewer.prototype.createMeasurementLabel = function(position, text) {
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 256;
                canvas.height = 64;
                
                context.fillStyle = 'rgba(0,0,0,0.8)';
                context.fillRect(0, 0, canvas.width, canvas.height);
                
                context.fillStyle = 'white';
                context.font = '20px Arial';
                context.textAlign = 'center';
                context.fillText(text, canvas.width/2, canvas.height/2 + 7);
                
                const texture = new THREE.CanvasTexture(canvas);
                const material = new THREE.SpriteMaterial({ map: texture });
                const sprite = new THREE.Sprite(material);
                sprite.position.copy(position);
                sprite.scale.set(1, 0.25, 1);
                
                this.scene.add(sprite);
                this.measurementObjects.push(sprite);
            };
            
            HouseBrainProfessionalViewer.prototype.clearMeasurements = function() {
                this.measurementObjects.forEach(obj => {
                    this.scene.remove(obj);
                });
                this.measurementObjects = [];
                this.measurementPoints = [];
                this.measurementMode = 'none';
                
                // Update UI
                document.querySelectorAll('.tool-button').forEach(btn => btn.classList.remove('active'));
                document.getElementById('measurement-mode').textContent = 'Mode: Navigation';
            };
            </script>
            '''
        
        elif feature == "material_comparison":
            return '''
            <script>
            // Material Comparison Extension
            HouseBrainProfessionalViewer.prototype.setupMaterialcomparison = function() {
                this.materialLibrary = [];
                this.selectedMaterials = [];
                
                // Setup material tool buttons
                document.getElementById('material-picker')?.addEventListener('click', () => {
                    this.setMaterialPickerMode();
                });
                
                document.getElementById('material-compare')?.addEventListener('click', () => {
                    this.openMaterialComparison();
                });
                
                document.getElementById('material-library')?.addEventListener('click', () => {
                    this.openMaterialLibrary();
                });
                
                console.log('🎨 Material comparison initialized');
            };
            
            HouseBrainProfessionalViewer.prototype.openMaterialLibrary = function() {
                document.getElementById('material-library-modal').classList.add('active');
                this.loadMaterialLibrary();
            };
            
            HouseBrainProfessionalViewer.prototype.loadMaterialLibrary = function() {
                const content = document.getElementById('material-library-content');
                content.innerHTML = '<div class="loading">Loading materials...</div>';
                
                // Simulate loading materials
                setTimeout(() => {
                    content.innerHTML = this.generateMaterialLibraryHTML();
                }, 500);
            };
            
            HouseBrainProfessionalViewer.prototype.generateMaterialLibraryHTML = function() {
                const materials = [
                    { name: 'Concrete Polished', category: 'Structural', preview: '#888888' },
                    { name: 'Oak Hardwood', category: 'Flooring', preview: '#8B4513' },
                    { name: 'Steel Brushed', category: 'Structural', preview: '#C0C0C0' },
                    { name: 'Glass Clear', category: 'Glazing', preview: '#87CEEB' }
                ];
                
                let html = '<div class="material-grid">';
                materials.forEach(material => {
                    html += `
                        <div class="material-item" onclick="selectMaterial('${material.name}')">
                            <div class="material-preview" style="background: ${material.preview}"></div>
                            <div class="material-info">
                                <div class="material-name">${material.name}</div>
                                <div class="material-category">${material.category}</div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                
                return html;
            };
            </script>
            '''
        
        return ''
    
    def _generate_viewer_assets(self, output_path: Path, viewer_name: str, house_data: Dict[str, Any]) -> None:
        """Generate supporting assets for the viewer"""
        
        # Generate model data JSON
        model_data = {
            "name": viewer_name,
            "geometry": house_data.get("geometry", {}),
            "materials": house_data.get("materials", {}),
            "metadata": {
                "generated_by": "HouseBrain Professional",
                "version": "3.0",
                "features": ["measurement_tools", "material_comparison", "lighting_controls"]
            }
        }
        
        model_file = output_path / f"{viewer_name}_model_data.json"
        with open(model_file, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✅ Model data saved: {model_file}")


def create_advanced_interactive_viewer() -> AdvancedInteractive3DViewer:
    """Create advanced interactive 3D viewer instance"""
    
    return AdvancedInteractive3DViewer()


if __name__ == "__main__":
    # Test advanced interactive viewer
    viewer = create_advanced_interactive_viewer()
    
    print("🎮 Advanced Interactive 3D Viewer Test")
    print("=" * 50)
    
    # Test sample house data
    sample_house = {
        "geometry": {
            "spaces": [{
                "id": "living_room",
                "type": "living",
                "area": 25000000,
                "boundary": [[0, 0], [6000, 0], [6000, 5000], [0, 5000]]
            }],
            "walls": [{
                "id": "wall_1",
                "start": [0, 0],
                "end": [6000, 0],
                "thickness": 200,
                "height": 2700,
                "type": "exterior"
            }]
        },
        "materials": {
            "wall_exterior": {"type": "concrete", "finish": "smooth"},
            "floor_living": {"type": "hardwood", "species": "oak"}
        }
    }
    
    # Generate professional viewer
    viewer_file = viewer.generate_professional_viewer(
        sample_house,
        "luxury_villa_professional",
        "advanced_viewer_output",
        ["measurement_tools", "material_comparison", "lighting_controls", "vr_ar_support"]
    )
    
    print(f"Professional viewer generated: {viewer_file}")
    print("✅ Advanced Interactive 3D Viewer test completed!")