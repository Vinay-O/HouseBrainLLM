# HouseBrain LLM — Project Brief

## Goal
Build a **custom LLM** ("HouseBrain") that:
- Takes structured user input (plot size, bedrooms, floors, budget, style, etc.).
- Deep-thinks like an **architect + civil engineer + interior designer**.
- Outputs:
  - **Engineering-grade 2D floor plans** (SVG)
  - **Engineering-grade 3D floor plans** (OBJ/GLB)
  - **Beautiful 3D elevations** (Blender render)
  - **Construction-worthy 3D model**
  - **Interior layouts**
  - **Construction cost estimates**
  - **Construction sequence/flow**

## Architecture
- LLM always returns **strict JSON** matching `schema.py`
- Schema uses `levels[]` for multi-floor support (unlimited floors)
- Validation checks: room areas, stair design, corridor widths, grid alignment, daylight, code compliance
- Blender facade style kit renders `axon.png`, `front_ortho.png`, `hero.png`

## Future Roadmap
- **Phase 1**: Residential houses (current)
- **Phase 2**: Indian mixed-use (res+commercial)
- **Phase 3**: Commercial-only buildings
- **Phase 4**: Apartments & high-rises
- **Phase 5**: Regional expansion — North America, Europe, Australia (with local building codes)

## Next Steps
1. Flesh out validators (corridors, stairs, headroom, egress, stacking).
2. Implement solver for room adjacency & routing.
3. Upgrade Blender kit to real window/door modules.
4. Generate synthetic dataset → fine-tune on DeepSeek via QLoRA.
5. Integrate with existing app to replace external AI calls.

## Integration Plan
- LLM service runs locally via Ollama (`deepseek-r1:8b`)
- API endpoint returns JSON + plan render
- App sends user input → HouseBrain LLM → receives JSON + images

## Technical Specifications

### Input Schema
- Plot dimensions and orientation
- Number of floors, bedrooms, bathrooms
- Architectural style preferences
- Budget constraints
- Room-specific requirements

### Output Schema
- Multi-level floor plans with room layouts
- Structural elements (walls, doors, windows, stairs)
- 3D geometry for visualization
- Construction specifications
- Cost estimates and material requirements

### Validation Rules
- Minimum room sizes and proportions
- Stair design and headroom requirements
- Corridor widths for accessibility
- Grid alignment for construction efficiency
- Daylight and ventilation requirements
- Building code compliance

### Rendering Pipeline
- SVG generation for 2D plans
- OBJ/GLB export for 3D models
- Blender integration for photorealistic renders
- Multiple view angles (front, side, isometric)
