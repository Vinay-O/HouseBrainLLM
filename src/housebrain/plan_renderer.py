from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple


@dataclass
class Wall:
    id: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    type: str  # exterior | interior
    thickness: float
    subtype: str | None = None  # plumbing | None


@dataclass
class Opening:
    id: str
    wall_id: str
    type: str  # door | window
    position: float  # 0..1 along the wall
    width: float
    metadata: Dict[str, Any]


@dataclass
class Space:
    id: str
    name: str
    type: str
    boundary: List[Tuple[float, float]]


class CADRenderer:
    def __init__(self, plan: Dict[str, Any], sheet_mode: str = "floor") -> None:
        self.units = plan.get("metadata", {}).get("units", "mm")
        self.scale = float(plan.get("metadata", {}).get("scale", 100))
        self.sheet_mode = sheet_mode  # floor | rcp | power | plumbing
        self.walls = [
            Wall(
                id=w["id"],
                start=tuple(w["start"]),
                end=tuple(w["end"]),
                type=w["type"],
                thickness=float(w.get("thickness", 115 if w["type"] == "interior" else 230)),
            )
            for w in plan.get("walls", [])
        ]
        self.openings = []
        for o in plan.get("openings", []):
            meta = dict(o.get("metadata", {}))
            # Carry v2 fields into metadata for unified access
            for k in [
                "handing",
                "swing",
                "window_operation",
                "height",
                "sill_height",
                "head_height",
                "frame_depth",
                "frame_width",
                "mullion_pattern",
            ]:
                if k in o:
                    meta[k] = o[k]
            self.openings.append(
                Opening(
                    id=o["id"],
                    wall_id=o["wall_id"],
                    type=o["type"],
                    position=float(o["position"]),
                    width=float(o["width"]),
                    metadata=meta,
                )
            )
        self.spaces = [
            Space(
                id=s["id"],
                name=s["name"],
                type=s.get("type", "room"),
                boundary=[tuple(p) for p in s.get("boundary", [])],
            )
            for s in plan.get("spaces", [])
        ]
        # Visual wall thickness in pixels (clamped, independent of scale)
        self.outer_wall_px = 8.0
        self.inner_wall_px = 5.0

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _add_fixture_symbols(self, svg: List[str], sp: Space, T) -> None:
        """Add appropriate fixture symbols based on room type and name."""
        if not sp.boundary or len(sp.boundary) < 3:
            return
            
        # Calculate room center and dimensions
        xs = [p[0] for p in sp.boundary]
        ys = [p[1] for p in sp.boundary]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w_mm = maxx - minx
        h_mm = maxy - miny
        
        name_lower = (sp.name or "").lower()
        space_type = (sp.type or "").lower()
        
        # Sheet mode gating
        if getattr(self, "sheet_mode", "floor") in ("rcp", "power"):
            # Electrical-only for RCP/Power
            self._add_electrical_symbols(svg, minx, miny, w_mm, h_mm, space_type, T)
            return

        if getattr(self, "sheet_mode", "floor") == "plumbing":
            # Only wet fixtures/utility
            if any(k in name_lower for k in ["bath", "toilet", "wash", "wc", "powder"]):
                self._add_bathroom_fixtures(svg, minx, miny, w_mm, h_mm, T)
            elif any(k in name_lower for k in ["utility", "laundry", "mechanical"]):
                self._add_utility_fixtures(svg, minx, miny, w_mm, h_mm, T)
            return

        # Default floor plan: draw fixtures and electrical
        if any(k in name_lower for k in ["bath", "toilet", "wash", "wc", "powder"]):
            self._add_bathroom_fixtures(svg, minx, miny, w_mm, h_mm, T)
        elif any(k in name_lower for k in ["kitchen", "kitchenette"]):
            self._add_kitchen_fixtures(svg, minx, miny, w_mm, h_mm, T)
        elif any(k in name_lower for k in ["living", "family", "great", "room"]):
            self._add_living_fixtures(svg, minx, miny, w_mm, h_mm, T)
        elif any(k in name_lower for k in ["bed", "master", "guest"]):
            self._add_bedroom_fixtures(svg, minx, miny, w_mm, h_mm, T)
        elif any(k in name_lower for k in ["utility", "laundry", "mechanical"]):
            self._add_utility_fixtures(svg, minx, miny, w_mm, h_mm, T)
        # Always add electrical in floor mode
        self._add_electrical_symbols(svg, minx, miny, w_mm, h_mm, space_type, T)

    def _add_bathroom_fixtures(self, svg: List[str], minx: float, miny: float, w_mm: float, h_mm: float, T) -> None:
        """Add professional bathroom fixtures with detailed symbols."""
        # TOILET - Standard 2-piece toilet with tank
        toilet_x = minx + 400  # 400mm from wall per code
        toilet_y = miny + 300
        TX, TY = T(toilet_x, toilet_y)
        # Toilet bowl (elongated)
        svg.append(f"<ellipse cx='{TX:.1f}' cy='{TY:.1f}' rx='18' ry='25' class='fixture' fill='white'/>")
        # Toilet tank
        svg.append(f"<rect x='{TX-12:.1f}' y='{TY-35:.1f}' width='24' height='20' class='fixture' fill='white' rx='3'/>")
        # Toilet seat (open)
        svg.append(f"<ellipse cx='{TX:.1f}' cy='{TY:.1f}' rx='15' ry='20' class='fixture' fill='none' stroke-width='2'/>")
        # Flush handle
        svg.append(f"<circle cx='{TX+10:.1f}' cy='{TY-25:.1f}' r='2' class='fixture'/>")
        
        # VANITY & SINK - Double sink if space allows
        if w_mm > h_mm:  # wider than tall
            vanity_x = minx + w_mm * 0.6
            vanity_y = miny + 150
            vanity_width = min(1200, w_mm * 0.3)  # Max 48" vanity
        else:  # taller than wide
            vanity_x = minx + 150
            vanity_y = miny + h_mm * 0.3
            vanity_width = min(900, w_mm * 0.4)
        
        VX, VY = T(vanity_x, vanity_y)
        # Vanity cabinet
        svg.append(f"<rect x='{VX-vanity_width/4:.1f}' y='{VY-15:.1f}' width='{vanity_width/2:.1f}' height='30' class='fixture' fill='#D2691E' opacity='0.3'/>")
        
        # Sinks - double if vanity > 900mm
        if vanity_width > 900:
            # Left sink
            svg.append(f"<ellipse cx='{VX-vanity_width/6:.1f}' cy='{VY:.1f}' rx='12' ry='8' class='fixture' fill='white'/>")
            svg.append(f"<circle cx='{VX-vanity_width/6:.1f}' cy='{VY-8:.1f}' r='3' class='fixture'/>")  # Faucet
            # Right sink
            svg.append(f"<ellipse cx='{VX+vanity_width/6:.1f}' cy='{VY:.1f}' rx='12' ry='8' class='fixture' fill='white'/>")
            svg.append(f"<circle cx='{VX+vanity_width/6:.1f}' cy='{VY-8:.1f}' r='3' class='fixture'/>")  # Faucet
        else:
            # Single sink
            svg.append(f"<ellipse cx='{VX:.1f}' cy='{VY:.1f}' rx='15' ry='10' class='fixture' fill='white'/>")
            svg.append(f"<circle cx='{VX:.1f}' cy='{VY-10:.1f}' r='3' class='fixture'/>")  # Faucet
        
        # Mirror above vanity
        svg.append(f"<rect x='{VX-vanity_width/4:.1f}' y='{VY-40:.1f}' width='{vanity_width/2:.1f}' height='20' fill='none' stroke='#666' stroke-width='1' stroke-dasharray='2,2'/>")
        
        # SHOWER/TUB based on room size
        if w_mm > 2000 and h_mm > 2000:  # Large bathroom - separate shower
            # Shower stall (36" x 36" minimum)
            shower_x = minx + w_mm - 500
            shower_y = miny + h_mm - 500
            ShX, ShY = T(shower_x, shower_y)
            # Shower base
            svg.append(f"<rect x='{ShX-45:.1f}' y='{ShY-45:.1f}' width='90' height='90' class='fixture' fill='none' stroke-width='2'/>")
            # Shower drain
            svg.append(f"<circle cx='{ShX:.1f}' cy='{ShY:.1f}' r='4' class='fixture'/>")
            # Shower head
            svg.append(f"<circle cx='{ShX:.1f}' cy='{ShY-35:.1f}' r='6' class='fixture' fill='none'/>")
            # Shower door swing
            svg.append(f"<path d='M {ShX-45:.1f} {ShY+45:.1f} A 90 90 0 0 1 {ShX+45:.1f} {ShY-45:.1f}' class='fixture' fill='none' stroke-dasharray='3,2'/>")
            
            # Separate tub if space allows
            if w_mm > 3000:
                tub_x = minx + 300
                tub_y = miny + h_mm - 400
                TubX, TubY = T(tub_x, tub_y)
                # Standard tub (60" x 30")
                svg.append(f"<rect x='{TubX:.1f}' y='{TubY-38:.1f}' width='150' height='75' class='fixture' fill='white' rx='8'/>")
                # Tub faucet
                svg.append(f"<circle cx='{TubX+20:.1f}' cy='{TubY-30:.1f}' r='4' class='fixture'/>")
                # Shower head above tub
                svg.append(f"<circle cx='{TubX+75:.1f}' cy='{TubY-50:.1f}' r='5' class='fixture' fill='none'/>")
        else:
            # Smaller bathroom - tub/shower combo
            tub_x = minx + w_mm - 400
            tub_y = miny + h_mm - 300
            TubX, TubY = T(tub_x, tub_y)
            # Tub/shower combo
            svg.append(f"<rect x='{TubX-75:.1f}' y='{TubY-38:.1f}' width='150' height='75' class='fixture' fill='white' rx='5'/>")
            # Shower curtain rod
            svg.append(f"<line x1='{TubX-75:.1f}' y1='{TubY-50:.1f}' x2='{TubX+75:.1f}' y2='{TubY-50:.1f}' class='fixture' stroke-dasharray='5,3'/>")
            # Faucet/shower controls
            svg.append(f"<circle cx='{TubX-60:.1f}' cy='{TubY-30:.1f}' r='4' class='fixture'/>")
            svg.append(f"<circle cx='{TubX+60:.1f}' cy='{TubY-45:.1f}' r='5' class='fixture' fill='none'/>")  # Shower head

    def _add_kitchen_fixtures(self, svg: List[str], minx: float, miny: float, w_mm: float, h_mm: float, T) -> None:
        """Add professional kitchen fixtures and appliances."""
        # KITCHEN ISLAND or PENINSULA (if space allows)
        if w_mm > 3000 and h_mm > 2500:  # Large kitchen gets island
            island_x = minx + w_mm * 0.5
            island_y = miny + h_mm * 0.6
            IX, IY = T(island_x, island_y)
            # Island base (4' x 8' typical)
            svg.append(f"<rect x='{IX-60:.1f}' y='{IY-40:.1f}' width='120' height='80' class='fixture' fill='#DEB887' opacity='0.4' rx='5'/>")
            # Island sink (if large enough)
            if w_mm > 4000:
                svg.append(f"<ellipse cx='{IX+30:.1f}' cy='{IY:.1f}' rx='15' ry='10' class='fixture' fill='white'/>")
                svg.append(f"<circle cx='{IX+30:.1f}' cy='{IY-8:.1f}' r='3' class='fixture'/>")  # Faucet
        
        # MAIN SINK (along perimeter)
        if w_mm > h_mm:  # Wider kitchen
            sink_x = minx + w_mm * 0.25
            sink_y = miny + 150  # Against wall
        else:
            sink_x = minx + 150
            sink_y = miny + h_mm * 0.25
        SX, SY = T(sink_x, sink_y)
        
        # Kitchen cabinet base
        svg.append(f"<rect x='{SX-40:.1f}' y='{SY-20:.1f}' width='80' height='40' class='fixture' fill='#D2691E' opacity='0.3'/>")
        # Double bowl sink
        svg.append(f"<ellipse cx='{SX-15:.1f}' cy='{SY:.1f}' rx='12' ry='8' class='fixture' fill='white'/>")
        svg.append(f"<ellipse cx='{SX+15:.1f}' cy='{SY:.1f}' rx='12' ry='8' class='fixture' fill='white'/>")
        # Faucet with sprayer
        svg.append(f"<circle cx='{SX:.1f}' cy='{SY-10:.1f}' r='3' class='fixture'/>")
        svg.append(f"<circle cx='{SX+8:.1f}' cy='{SY-8:.1f}' r='2' class='fixture'/>")  # Sprayer
        
        # RANGE/COOKTOP (36" wide professional)
        range_x = minx + w_mm * 0.7
        range_y = miny + 150
        RX, RY = T(range_x, range_y)
        # Range base
        svg.append(f"<rect x='{RX-45:.1f}' y='{RY-30:.1f}' width='90' height='60' class='fixture' fill='#333' opacity='0.8'/>")
        # 6 burners (professional range)
        for i, (dx, dy) in enumerate([(-25, -15), (0, -15), (25, -15), (-25, 15), (0, 15), (25, 15)]):
            svg.append(f"<circle cx='{RX+dx:.1f}' cy='{RY+dy:.1f}' r='6' class='fixture' fill='none' stroke-width='2'/>")
        # Range hood above (outline)
        svg.append(f"<rect x='{RX-50:.1f}' y='{RY-50:.1f}' width='100' height='15' fill='none' stroke='#666' stroke-width='1' stroke-dasharray='3,2'/>")
        
        # REFRIGERATOR (counter depth)
        fridge_x = minx + w_mm - 300
        fridge_y = miny + 200
        FX, FY = T(fridge_x, fridge_y)
        svg.append(f"<rect x='{FX-40:.1f}' y='{FY-30:.1f}' width='80' height='60' class='fixture' fill='white' stroke='#333' stroke-width='2'/>")
        # French doors
        svg.append(f"<line x1='{FX:.1f}' y1='{FY-30:.1f}' x2='{FX:.1f}' y2='{FY+30:.1f}' stroke='#333' stroke-width='1'/>")
        # Handles
        svg.append(f"<circle cx='{FX-8:.1f}' cy='{FY:.1f}' r='2' class='fixture'/>")
        svg.append(f"<circle cx='{FX+8:.1f}' cy='{FY:.1f}' r='2' class='fixture'/>")
        
        # DISHWASHER (next to sink)
        dw_x = sink_x + 600  # 24" from sink
        dw_y = sink_y
        DX, DY = T(dw_x, dw_y)
        svg.append(f"<rect x='{DX-30:.1f}' y='{DY-15:.1f}' width='60' height='30' class='fixture' fill='#E6E6FA' stroke='#333'/>")
        svg.append(f"<text x='{DX:.1f}' y='{DY:.1f}' class='sub' text-anchor='middle' font-size='8'>DW</text>")
        
        # MICROWAVE (above range or island)
        if w_mm > 3500:  # Built-in microwave
            mw_x = range_x
            mw_y = range_y - 400
            MX, MY = T(mw_x, mw_y)
            svg.append(f"<rect x='{MX-35:.1f}' y='{MY-12:.1f}' width='70' height='24' fill='none' stroke='#666' stroke-dasharray='2,2'/>")
            svg.append(f"<text x='{MX:.1f}' y='{MY:.1f}' class='sub' text-anchor='middle' font-size='6'>MW</text>")
        
        # UPPER CABINETS (dashed outline)
        cabinet_y = miny + 50  # Above counter
        for x_pos in [0.2, 0.4, 0.6, 0.8]:
            if x_pos * w_mm < w_mm - 600:  # Don't overlap with fridge
                CX, CY = T(minx + x_pos * w_mm, cabinet_y)
                svg.append(f"<rect x='{CX-30:.1f}' y='{CY-15:.1f}' width='60' height='30' fill='none' stroke='#999' stroke-width='1' stroke-dasharray='4,3'/>")
        
        # PANTRY (if corner space available)
        if w_mm > 2500 and h_mm > 2500:
            pantry_x = minx + w_mm - 200
            pantry_y = miny + h_mm - 400
            PX, PY = T(pantry_x, pantry_y)
            svg.append(f"<rect x='{PX-25:.1f}' y='{PY-40:.1f}' width='50' height='80' class='fixture' fill='#DEB887' opacity='0.3'/>")
            svg.append(f"<text x='{PX:.1f}' y='{PY:.1f}' class='sub' text-anchor='middle' font-size='8'>PANTRY</text>")

    def _add_living_fixtures(self, svg: List[str], minx: float, miny: float, w_mm: float, h_mm: float, T) -> None:
        """Add electrical outlets and switches."""
        # Electrical outlets along walls (every 3-4m)
        outlets_x = []
        outlets_y = []
        
        # Along horizontal walls
        for i in range(1, int(w_mm / 3000) + 1):
            x = minx + i * 3000
            if x < minx + w_mm - 500:
                outlets_x.extend([x, x])
                outlets_y.extend([miny + 100, miny + h_mm - 100])
        
        # Along vertical walls  
        for i in range(1, int(h_mm / 3000) + 1):
            y = miny + i * 3000
            if y < miny + h_mm - 500:
                outlets_x.extend([minx + 100, minx + w_mm - 100])
                outlets_y.extend([y, y])
        
        for ox, oy in zip(outlets_x, outlets_y):
            OX, OY = T(ox, oy)
            svg.append(f"<rect x='{OX-4:.1f}' y='{OY-3:.1f}' width='8' height='6' class='electrical'/>")
            svg.append(f"<circle cx='{OX-2:.1f}' cy='{OY:.1f}' r='1' class='electrical'/>")
            svg.append(f"<circle cx='{OX+2:.1f}' cy='{OY:.1f}' r='1' class='electrical'/>")

    def _add_bedroom_fixtures(self, svg: List[str], minx: float, miny: float, w_mm: float, h_mm: float, T) -> None:
        """Add bedroom fixtures - outlets and overhead lighting."""
        # Ceiling light (center)
        cx, cy = minx + w_mm/2, miny + h_mm/2
        CX, CY = T(cx, cy)
        svg.append(f"<circle cx='{CX:.1f}' cy='{CY:.1f}' r='8' class='electrical' fill='none'/>")
        svg.append(f"<line x1='{CX-6:.1f}' y1='{CY:.1f}' x2='{CX+6:.1f}' y2='{CY:.1f}' class='electrical'/>")
        svg.append(f"<line x1='{CX:.1f}' y1='{CY-6:.1f}' x2='{CX:.1f}' y2='{CY+6:.1f}' class='electrical'/>")
        
        # Wall outlets (bedside)
        for x_pos in [0.25, 0.75]:
            ox = minx + w_mm * x_pos
            oy = miny + 100  # Near headboard wall
            OX, OY = T(ox, oy)
            svg.append(f"<rect x='{OX-4:.1f}' y='{OY-3:.1f}' width='8' height='6' class='electrical'/>")
            svg.append(f"<circle cx='{OX-2:.1f}' cy='{OY:.1f}' r='1' class='electrical'/>")
            svg.append(f"<circle cx='{OX+2:.1f}' cy='{OY:.1f}' r='1' class='electrical'/>")

    def _add_utility_fixtures(self, svg: List[str], minx: float, miny: float, w_mm: float, h_mm: float, T) -> None:
        """Add utility room fixtures."""
        # Water heater
        wh_x = minx + 300
        wh_y = miny + 300
        WHX, WHY = T(wh_x, wh_y)
        svg.append(f"<circle cx='{WHX:.1f}' cy='{WHY:.1f}' r='20' class='fixture'/>")
        svg.append(f"<text x='{WHX:.1f}' y='{WHY+3:.1f}' class='electrical' text-anchor='middle' font-size='8'>WH</text>")
        
        # Washer/Dryer connections if space allows
        if w_mm > 2000:
            # Washer connection
            w_x = minx + w_mm - 600
            w_y = miny + 200
            WX, WY = T(w_x, w_y)
            svg.append(f"<rect x='{WX-15:.1f}' y='{WY-15:.1f}' width='30' height='30' class='fixture'/>")
            svg.append(f"<text x='{WX:.1f}' y='{WY+3:.1f}' class='electrical' text-anchor='middle' font-size='6'>W</text>")
            
            # Dryer connection
            d_x = w_x + 300
            d_y = w_y
            DX, DY = T(d_x, d_y)
            svg.append(f"<rect x='{DX-15:.1f}' y='{DY-15:.1f}' width='30' height='30' class='fixture'/>")
            svg.append(f"<text x='{DX:.1f}' y='{DY+3:.1f}' class='electrical' text-anchor='middle' font-size='6'>D</text>")

    def _add_electrical_symbols(self, svg: List[str], minx: float, miny: float, w_mm: float, h_mm: float, space_type: str, T) -> None:
        """Add comprehensive electrical symbols based on space type and building codes."""
        
        # OUTLETS - Code requires outlets every 3.6m (12') max spacing
        outlet_spacing = 3600  # mm
        outlets = []
        
        # Along horizontal walls
        for x in range(int(minx + 600), int(minx + w_mm - 600), outlet_spacing):
            outlets.extend([(x, miny + 150), (x, miny + h_mm - 150)])  # Top and bottom walls
        
        # Along vertical walls  
        for y in range(int(miny + 600), int(miny + h_mm - 600), outlet_spacing):
            outlets.extend([(minx + 150, y), (minx + w_mm - 150, y)])  # Left and right walls
        
        # Draw standard duplex outlets
        for x, y in outlets:
            OX, OY = T(x, y)
            svg.append(f"<circle cx='{OX:.1f}' cy='{OY:.1f}' r='4' class='electrical' fill='white'/>")
            svg.append(f"<line x1='{OX-2:.1f}' y1='{OY-1:.1f}' x2='{OX+2:.1f}' y2='{OY-1:.1f}' class='electrical'/>")
            svg.append(f"<line x1='{OX-2:.1f}' y1='{OY+1:.1f}' x2='{OX+2:.1f}' y2='{OY+1:.1f}' class='electrical'/>")
        
        # SWITCHES - Near door openings
        switch_x = minx + 300  # 12" from door frame
        switch_y = miny + 200
        SX, SY = T(switch_x, switch_y)
        svg.append(f"<rect x='{SX-3:.1f}' y='{SY-6:.1f}' width='6' height='12' class='electrical' fill='white'/>")
        svg.append(f"<text x='{SX:.1f}' y='{SY:.1f}' class='electrical' text-anchor='middle' font-size='6'>S</text>")
        
        # LIGHTING based on space type
        if space_type in ["bathroom", "kitchen"]:
            # Recessed lights every 1.5m
            for lx in range(int(minx + 750), int(minx + w_mm - 750), 1500):
                for ly in range(int(miny + 750), int(miny + h_mm - 750), 1500):
                    LX, LY = T(lx, ly)
                    svg.append(f"<circle cx='{LX:.1f}' cy='{LY:.1f}' r='6' class='electrical' fill='none' stroke-dasharray='2,1'/>")
                    svg.append(f"<text x='{LX:.1f}' y='{LY+2:.1f}' class='electrical' text-anchor='middle' font-size='4'>R</text>")
        
        elif space_type in ["living", "dining", "bedroom"]:
            # Central ceiling fixture
            center_x, center_y = minx + w_mm/2, miny + h_mm/2
            CX, CY = T(center_x, center_y)
            svg.append(f"<circle cx='{CX:.1f}' cy='{CY:.1f}' r='8' class='electrical' fill='none'/>")
            svg.append(f"<line x1='{CX-6:.1f}' y1='{CY:.1f}' x2='{CX+6:.1f}' y2='{CY:.1f}' class='electrical'/>")
            svg.append(f"<line x1='{CX:.1f}' y1='{CY-6:.1f}' x2='{CX:.1f}' y2='{CY+6:.1f}' class='electrical'/>")
        
        # SPECIAL OUTLETS based on space type
        if space_type == "kitchen":
            # GFCI outlets above counter
            for gx in [minx + w_mm * 0.25, minx + w_mm * 0.75]:
                GX, GY = T(gx, miny + 100)
                svg.append(f"<rect x='{GX-4:.1f}' y='{GY-4:.1f}' width='8' height='8' class='electrical' fill='white'/>")
                svg.append(f"<text x='{GX:.1f}' y='{GY+1:.1f}' class='electrical' text-anchor='middle' font-size='4'>G</text>")
            
            # 220V outlet for range
            range_outlet_x, range_outlet_y = minx + w_mm * 0.7, miny + 50
            RGX, RGY = T(range_outlet_x, range_outlet_y)
            svg.append(f"<rect x='{RGX-5:.1f}' y='{RGY-5:.1f}' width='10' height='10' class='electrical' fill='yellow'/>")
            svg.append(f"<text x='{RGX:.1f}' y='{RGY+1:.1f}' class='electrical' text-anchor='middle' font-size='4'>220</text>")
        
        elif space_type == "bathroom":
            # GFCI outlets required
            gfci_x, gfci_y = minx + w_mm * 0.7, miny + 150
            GX, GY = T(gfci_x, gfci_y)
            svg.append(f"<rect x='{GX-4:.1f}' y='{GY-4:.1f}' width='8' height='8' class='electrical' fill='white'/>")
            svg.append(f"<text x='{GX:.1f}' y='{GY+1:.1f}' class='electrical' text-anchor='middle' font-size='4'>G</text>")
            
            # Exhaust fan
            fan_x, fan_y = minx + w_mm/2, miny + h_mm/2
            FX, FY = T(fan_x, fan_y)
            svg.append(f"<circle cx='{FX:.1f}' cy='{FY:.1f}' r='8' class='electrical' fill='none'/>")
            svg.append(f"<text x='{FX:.1f}' y='{FY+2:.1f}' class='electrical' text-anchor='middle' font-size='6'>EF</text>")
        
        elif space_type == "utility":
            # 220V dryer outlet
            dryer_x, dryer_y = minx + 200, miny + 200
            DX, DY = T(dryer_x, dryer_y)
            svg.append(f"<rect x='{DX-5:.1f}' y='{DY-5:.1f}' width='10' height='10' class='electrical' fill='yellow'/>")
            svg.append(f"<text x='{DX:.1f}' y='{DY+1:.1f}' class='electrical' text-anchor='middle' font-size='4'>220</text>")
        
        # SMOKE DETECTORS (required in all spaces per code)
        smoke_x, smoke_y = minx + w_mm * 0.8, miny + h_mm * 0.2
        SMX, SMY = T(smoke_x, smoke_y)
        svg.append(f"<circle cx='{SMX:.1f}' cy='{SMY:.1f}' r='6' class='electrical' fill='red' opacity='0.3'/>")
        svg.append(f"<text x='{SMX:.1f}' y='{SMY+2:.1f}' class='electrical' text-anchor='middle' font-size='6'>SD</text>")

    def _add_chained_dimensions(self, svg: List[str], T, width: int, height: int, margin: int) -> None:
        """Add chained dimensioning system for room layouts."""
        if not self.spaces:
            return
            
        # Get building bounds
        all_points = []
        for sp in self.spaces:
            all_points.extend(sp.boundary)
        if not all_points:
            return
            
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        
        # HORIZONTAL CHAINED DIMENSIONS (bottom of plan)
        # Find all unique X coordinates of walls/openings
        x_points = set()
        for wall in self.walls:
            x_points.add(wall.start[0])
            x_points.add(wall.end[0])
        x_coords = sorted(x_points)
        
        # Draw chained dimensions
        y_dim_line = miny - 600  # Below the building
        TY_DIM = T(0, y_dim_line)[1]
        
        for i in range(len(x_coords) - 1):
            x1, x2 = x_coords[i], x_coords[i + 1]
            TX1, _ = T(x1, 0)
            TX2, _ = T(x2, 0)
            
            # Dimension line segment
            svg.append(f"<line x1='{TX1:.1f}' y1='{TY_DIM:.1f}' x2='{TX2:.1f}' y2='{TY_DIM:.1f}' class='dim'/>")
            # Extension lines
            svg.append(f"<line x1='{TX1:.1f}' y1='{T(x1, miny)[1]:.1f}' x2='{TX1:.1f}' y2='{TY_DIM - 10:.1f}' class='dim'/>")
            svg.append(f"<line x1='{TX2:.1f}' y1='{T(x2, miny)[1]:.1f}' x2='{TX2:.1f}' y2='{TY_DIM - 10:.1f}' class='dim'/>")
            # Architectural ticks
            svg.append(f"<line x1='{TX1-4:.1f}' y1='{TY_DIM-4:.1f}' x2='{TX1+4:.1f}' y2='{TY_DIM+4:.1f}' class='tick'/>")
            svg.append(f"<line x1='{TX2-4:.1f}' y1='{TY_DIM-4:.1f}' x2='{TX2+4:.1f}' y2='{TY_DIM+4:.1f}' class='tick'/>")
            # Dimension text
            dim_mm = x2 - x1
            dim_text = self._mm_to_feet_inches_str(dim_mm)
            text_x = (TX1 + TX2) / 2
            svg.append(f"<text x='{text_x:.1f}' y='{TY_DIM - 15:.1f}' class='dim' text-anchor='middle' font-size='10'>{dim_text}</text>")
        
        # VERTICAL CHAINED DIMENSIONS (right side of plan)
        y_points = set()
        for wall in self.walls:
            y_points.add(wall.start[1])
            y_points.add(wall.end[1])
        y_coords = sorted(y_points)
        
        x_dim_line = maxx + 600  # Right of the building
        TX_DIM = T(x_dim_line, 0)[0]
        
        for i in range(len(y_coords) - 1):
            y1, y2 = y_coords[i], y_coords[i + 1]
            _, TY1 = T(0, y1)
            _, TY2 = T(0, y2)
            
            # Dimension line segment
            svg.append(f"<line x1='{TX_DIM:.1f}' y1='{TY1:.1f}' x2='{TX_DIM:.1f}' y2='{TY2:.1f}' class='dim'/>")
            # Extension lines
            svg.append(f"<line x1='{T(maxx, y1)[0]:.1f}' y1='{TY1:.1f}' x2='{TX_DIM + 10:.1f}' y2='{TY1:.1f}' class='dim'/>")
            svg.append(f"<line x1='{T(maxx, y2)[0]:.1f}' y1='{TY2:.1f}' x2='{TX_DIM + 10:.1f}' y2='{TY2:.1f}' class='dim'/>")
            # Architectural ticks
            svg.append(f"<line x1='{TX_DIM-4:.1f}' y1='{TY1-4:.1f}' x2='{TX_DIM+4:.1f}' y2='{TY1+4:.1f}' class='tick'/>")
            svg.append(f"<line x1='{TX_DIM-4:.1f}' y1='{TY2-4:.1f}' x2='{TX_DIM+4:.1f}' y2='{TY2+4:.1f}' class='tick'/>")
            # Dimension text
            dim_mm = y2 - y1
            dim_text = self._mm_to_feet_inches_str(dim_mm)
            text_y = (TY1 + TY2) / 2
            svg.append(f"<text x='{TX_DIM + 25:.1f}' y='{text_y:.1f}' class='dim' text-anchor='middle' font-size='10' transform='rotate(90 {TX_DIM + 25:.1f} {text_y:.1f})'>{dim_text}</text>")

    def _add_elevation_markers(self, svg: List[str], T, width: int, height: int, margin: int) -> None:
        """Add elevation markers showing floor heights and level changes."""
        if not self.spaces:
            return
            
        # Standard residential elevations
        elevations = {
            "basement": {"height": -8, "label": "B1", "color": "#666"},
            "ground": {"height": 0, "label": "01", "color": "#000"},
            "second": {"height": 108, "label": "02", "color": "#000"},  # 9' ceiling
            "attic": {"height": 216, "label": "AT", "color": "#999"}
        }
        
        # Place elevation markers in top-left corner
        start_x, start_y = margin + 20, margin + 60
        
        for i, (level, data) in enumerate(elevations.items()):
            marker_x = start_x
            marker_y = start_y + i * 25
            
            # Elevation circle
            svg.append(f"<circle cx='{marker_x:.1f}' cy='{marker_y:.1f}' r='12' fill='white' stroke='{data['color']}' stroke-width='2'/>")
            svg.append(f"<text x='{marker_x:.1f}' y='{marker_y + 3:.1f}' class='dim' text-anchor='middle' font-size='8' fill='{data['color']}'>{data['label']}</text>")
            
            # Elevation height
            svg.append(f"<text x='{marker_x + 20:.1f}' y='{marker_y + 3:.1f}' class='dim' font-size='8' fill='{data['color']}'>{data['height']:.0f}\"</text>")
            
            # Level line (if applicable to current floor)
            if level == "ground":  # Main floor
                # Draw level line across the plan
                all_points = []
                for sp in self.spaces:
                    all_points.extend(sp.boundary)
                xs = [p[0] for p in all_points]
                minx, maxx = min(xs), max(xs)
                
                level_y = margin + 40
                TX1, _ = T(minx, 0)
                TX2, _ = T(maxx, 0)
                svg.append(f"<line x1='{TX1:.1f}' y1='{level_y:.1f}' x2='{TX2:.1f}' y2='{level_y:.1f}' stroke='{data['color']}' stroke-width='1' stroke-dasharray='10,5'/>")

    def _add_radial_dimensions(self, svg: List[str], T) -> None:
        """Add radial dimensions for curved walls and arcs."""
        # Find curved walls or circular elements
        for wall in self.walls:
            # Check if wall has curvature (for future curved wall support)
            # For now, add example radial dimension for any corner > 90 degrees
            pass
        
        # Add radial dimensions for curved openings (bay windows, etc.)
        for opening in self.openings:
            if opening.metadata.get("style") == "bay_window":
                # Get the wall this opening is on
                wall = next((w for w in self.walls if w.id == opening.wall_id), None)
                if wall:
                    # Calculate opening position
                    x1, y1 = wall.start
                    x2, y2 = wall.end
                    op_x = x1 + opening.position * (x2 - x1)
                    op_y = y1 + opening.position * (y2 - y1)
                    
                    # Bay window projection (typical 24")
                    bay_depth = 600  # mm
                    wall_angle = math.atan2(y2 - y1, x2 - x1)
                    normal_angle = wall_angle + math.pi/2
                    
                    # Center of bay
                    center_x = op_x + bay_depth * math.cos(normal_angle)
                    center_y = op_y + bay_depth * math.sin(normal_angle)
                    CX, CY = T(center_x, center_y)
                    
                    # Radial dimension
                    radius_px = bay_depth * self.scale / 100  # Convert to pixels
                    svg.append(f"<circle cx='{CX:.1f}' cy='{CY:.1f}' r='{radius_px:.1f}' fill='none' stroke='#999' stroke-width='0.5' stroke-dasharray='3,2'/>")
                    
                    # Dimension line
                    end_x = CX + radius_px * 0.7
                    end_y = CY - radius_px * 0.7
                    svg.append(f"<line x1='{CX:.1f}' y1='{CY:.1f}' x2='{end_x:.1f}' y2='{end_y:.1f}' class='dim'/>")
                    
                    # Radius text
                    radius_text = self._mm_to_feet_inches_str(bay_depth)
                    svg.append(f"<text x='{end_x + 10:.1f}' y='{end_y - 5:.1f}' class='dim' font-size='8'>R={radius_text}</text>")

    def _mm_to_feet_inches_str(self, mm: float) -> str:
        """Convert millimeters to feet and inches string format."""
        inches = mm / 25.4
        feet = int(inches // 12)
        remaining_inches = inches % 12
        
        if feet > 0:
            return f"{feet}'-{remaining_inches:.0f}\""
        else:
            return f"{remaining_inches:.1f}\""

    def _add_architectural_grid(self, svg: List[str], T, width: int, height: int) -> None:
        """Add professional architectural grid system with major and minor lines."""
        if not self.spaces:
            return
            
        # Calculate building extents in world coordinates (mm)
        all_points = []
        for sp in self.spaces:
            all_points.extend(sp.boundary)
        for wll in self.walls:
            all_points.extend([wll.start, wll.end])
            
        if not all_points:
            return
            
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        # Grid spacing in mm (architectural standard)
        major_grid_mm = 6000  # 6m major grid (structural grid)
        minor_grid_mm = 1200  # 1.2m minor grid (layout grid)
        
        # Extend grid beyond building by one major module
        grid_min_x = (min_x // major_grid_mm - 1) * major_grid_mm
        grid_max_x = (max_x // major_grid_mm + 2) * major_grid_mm
        grid_min_y = (min_y // major_grid_mm - 1) * major_grid_mm
        grid_max_y = (max_y // major_grid_mm + 2) * major_grid_mm
        
        # Draw minor grid lines first (lighter)
        x = grid_min_x
        while x <= grid_max_x:
            if x % major_grid_mm != 0:  # Skip major grid positions
                X1, Y1 = T(x, grid_min_y)
                X2, Y2 = T(x, grid_max_y)
                if 0 <= X1 <= width:  # Only draw if visible
                    svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' class='grid-minor'/>")
            x += minor_grid_mm
            
        y = grid_min_y
        while y <= grid_max_y:
            if y % major_grid_mm != 0:  # Skip major grid positions
                X1, Y1 = T(grid_min_x, y)
                X2, Y2 = T(grid_max_x, y)
                if 0 <= Y1 <= height:  # Only draw if visible
                    svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' class='grid-minor'/>")
            y += minor_grid_mm
        
        # Draw major grid lines (structural grid)
        grid_labels_x = []
        grid_labels_y = []
        
        x = grid_min_x
        grid_letter = ord('A')
        while x <= grid_max_x:
            X1, Y1 = T(x, grid_min_y)
            X2, Y2 = T(x, grid_max_y)
            if 0 <= X1 <= width:
                svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' class='grid-major'/>")
                # Grid bubble at top and bottom
                if Y1 >= 0:
                    svg.append(f"<circle cx='{X1:.1f}' cy='{Y1-15:.1f}' r='10' fill='white' stroke='#666' stroke-width='1'/>")
                    svg.append(f"<text x='{X1:.1f}' y='{Y1-10:.1f}' class='label' text-anchor='middle' font-size='8'>{chr(grid_letter)}</text>")
                if Y2 <= height:
                    svg.append(f"<circle cx='{X2:.1f}' cy='{Y2+15:.1f}' r='10' fill='white' stroke='#666' stroke-width='1'/>")
                    svg.append(f"<text x='{X2:.1f}' y='{Y2+20:.1f}' class='label' text-anchor='middle' font-size='8'>{chr(grid_letter)}</text>")
                grid_letter += 1
            x += major_grid_mm
            
        y = grid_min_y
        grid_number = 1
        while y <= grid_max_y:
            X1, Y1 = T(grid_min_x, y)
            X2, Y2 = T(grid_max_x, y)
            if 0 <= Y1 <= height:
                svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y1:.1f}' class='grid-major'/>")
                # Grid bubble at left and right
                if X1 >= 0:
                    svg.append(f"<circle cx='{X1-15:.1f}' cy='{Y1:.1f}' r='10' fill='white' stroke='#666' stroke-width='1'/>")
                    svg.append(f"<text x='{X1-15:.1f}' y='{Y1+3:.1f}' class='label' text-anchor='middle' font-size='8'>{grid_number}</text>")
                if X2 <= width:
                    svg.append(f"<circle cx='{X2+15:.1f}' cy='{Y2:.1f}' r='10' fill='white' stroke='#666' stroke-width='1'/>")
                    svg.append(f"<text x='{X2+15:.1f}' y='{Y2+3:.1f}' class='label' text-anchor='middle' font-size='8'>{grid_number}</text>")
                grid_number += 1
            y += major_grid_mm
        
        # Add grid legend/note
        grid_legend_x, grid_legend_y = width - 200, 50
        svg.append(f"<g id='grid-legend'>")
        svg.append(f"<rect x='{grid_legend_x-10}' y='{grid_legend_y-10}' width='180' height='60' fill='white' stroke='#666' stroke-width='1' opacity='0.9'/>")
        svg.append(f"<text x='{grid_legend_x}' y='{grid_legend_y+5}' class='label' font-size='9' font-weight='bold'>STRUCTURAL GRID</text>")
        svg.append(f"<line x1='{grid_legend_x}' y1='{grid_legend_y+12}' x2='{grid_legend_x+30}' y2='{grid_legend_y+12}' class='grid-major'/>")
        svg.append(f"<text x='{grid_legend_x+35}' y='{grid_legend_y+16}' class='sub' font-size='7'>Major: 6.0m</text>")
        svg.append(f"<line x1='{grid_legend_x}' y1='{grid_legend_y+25}' x2='{grid_legend_x+30}' y2='{grid_legend_y+25}' class='grid-minor'/>")
        svg.append(f"<text x='{grid_legend_x+35}' y='{grid_legend_y+29}' class='sub' font-size='7'>Minor: 1.2m</text>")
        svg.append(f"<text x='{grid_legend_x}' y='{grid_legend_y+42}' class='sub' font-size='6'>Grid: A-1, B-2, C-3...</text>")
        svg.append(f"</g>")
        
        # Add material legend
        mat_legend_x, mat_legend_y = width - 200, 140
        svg.append(f"<g id='material-legend'>")
        svg.append(f"<rect x='{mat_legend_x-10}' y='{mat_legend_y-10}' width='180' height='85' fill='white' stroke='#666' stroke-width='1' opacity='0.9'/>")
        svg.append(f"<text x='{mat_legend_x}' y='{mat_legend_y+5}' class='label' font-size='9' font-weight='bold'>MATERIALS</text>")
        
        # Material samples
        sample_size = 12
        svg.append(f"<rect x='{mat_legend_x}' y='{mat_legend_y+12}' width='{sample_size}' height='{sample_size}' fill='url(#masonry)' stroke='#666' stroke-width='0.5'/>")
        svg.append(f"<text x='{mat_legend_x+18}' y='{mat_legend_y+21}' class='sub' font-size='7'>Masonry/Brick</text>")
        
        svg.append(f"<rect x='{mat_legend_x}' y='{mat_legend_y+26}' width='{sample_size}' height='{sample_size}' fill='url(#concrete)' stroke='#666' stroke-width='0.5'/>")
        svg.append(f"<text x='{mat_legend_x+18}' y='{mat_legend_y+35}' class='sub' font-size='7'>Concrete</text>")
        
        svg.append(f"<rect x='{mat_legend_x}' y='{mat_legend_y+40}' width='{sample_size}' height='{sample_size}' fill='url(#wood-grain)' stroke='#666' stroke-width='0.5'/>")
        svg.append(f"<text x='{mat_legend_x+18}' y='{mat_legend_y+49}' class='sub' font-size='7'>Wood Frame</text>")
        
        svg.append(f"<rect x='{mat_legend_x}' y='{mat_legend_y+54}' width='{sample_size}' height='{sample_size}' fill='url(#tile)' stroke='#666' stroke-width='0.5'/>")
        svg.append(f"<text x='{mat_legend_x+18}' y='{mat_legend_y+63}' class='sub' font-size='7'>Ceramic Tile</text>")
        
        svg.append(f"</g>")

    def _line_dir(self, a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]:
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        L = math.hypot(dx, dy)
        if L == 0:
            return 0.0, 0.0, 0.0
        return dx / L, dy / L, L

    def _wall_strip(self, a: Tuple[float, float], b: Tuple[float, float], t: float) -> str:
        ux, uy, L = self._line_dir(a, b)
        if L == 0:
            return ""
        px, py = -uy, ux
        off = t / 2.0
        ax = a[0] + px * off
        ay = a[1] + py * off
        bx = b[0] + px * off
        by = b[1] + py * off
        cx = b[0] - px * off
        cy = b[1] - py * off
        dx = a[0] - px * off
        dy = a[1] - py * off
        return f"M {ax:.1f} {ay:.1f} L {bx:.1f} {by:.1f} L {cx:.1f} {cy:.1f} L {dx:.1f} {dy:.1f} Z"

    def _opening_span(self, wall: Wall, opening: Opening) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        ux, uy, L = self._line_dir(wall.start, wall.end)
        oL = opening.width
        # position along wall measured from start
        s = opening.position * L
        cx = wall.start[0] + ux * s
        cy = wall.start[1] + uy * s
        ax = cx - ux * (oL / 2)
        ay = cy - uy * (oL / 2)
        bx = cx + ux * (oL / 2)
        by = cy + uy * (oL / 2)
        return (ax, ay), (bx, by)

    def render(self, width: int = 1800, height: int = 1200) -> str:
        svg = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<defs>",
            "<style>",
            "/* Professional Line Weight Hierarchy (CAD Standard) */",
            ".wall-cut { fill: #000; opacity: 1.0; stroke: #000; stroke-width: 3.5; }",
            ".wall-exterior { fill: #111; opacity: 1.0; stroke: #000; stroke-width: 2.5; }",
            ".wall-interior { fill: #111; opacity: 1.0; stroke: #000; stroke-width: 1.8; }",
            ".wall { fill: #111; opacity: 1.0; stroke: #000; stroke-width: 1.8; }",
            ".wall-plumbing { fill: #222; opacity: 1.0; stroke: #0aa; stroke-width: 1.8; stroke-dasharray: 3,2; }",
            ".wall-hidden { fill: none; stroke: #888; stroke-width: 0.8; stroke-dasharray: 5,3; }",
            ".door { stroke: #8B4513; stroke-width: 1.2; fill: none; }",
            ".door-swing { stroke: #8B4513; stroke-width: 0.8; fill: none; stroke-dasharray: 3,1; }",
            ".window { stroke: #0066CC; stroke-width: 1.2; fill: white; stroke-opacity: 0.9; }",
            ".window-sill { stroke: #0066CC; stroke-width: 2.0; fill: none; }",
            ".fixture { stroke: #333; stroke-width: 1.0; fill: none; }",
            ".electrical { stroke: #666; stroke-width: 0.8; fill: none; }",
            ".label { font-family: Arial, Helvetica, sans-serif; font-size: 11px; fill: #111; }",
            ".dim { stroke: #111; stroke-width: 0.75; fill: none; }",
            ".dimtxt { font-family: Arial, Helvetica, sans-serif; font-size: 10px; fill: #111; paint-order: stroke fill; stroke: #fff; stroke-width: 2px; }",
            ".tick { stroke: #111; stroke-width: 1.2; fill: none; }",
            ".cap { stroke: #111; stroke-width: 1.5; fill: none; }",
            ".pl-center { stroke: #0aa; stroke-width: 0.8; stroke-dasharray: 3,2; fill: none; }",
            ".grid-major { stroke: #ccc; stroke-width: 0.5; fill: none; }",
            ".grid-minor { stroke: #eee; stroke-width: 0.3; fill: none; }",
            ".title { font-family: Arial, Helvetica, sans-serif; font-size: 18px; font-weight: bold; fill: #000; }",
            ".sub { font-family: Arial, Helvetica, sans-serif; font-size: 11px; fill: #444; }",
            ".roomfill { fill: #F8F8F8; }",
            "</style>",
            "<marker id='dim-arrow' markerWidth='8' markerHeight='8' refX='4' refY='4' orient='auto-start-reverse' markerUnits='strokeWidth'>",
            "<path d='M 0 0 L 8 4 L 0 8 z' fill='#111'/>",
            "</marker>",
            "<pattern id='tile' width='14' height='14' patternUnits='userSpaceOnUse'>",
            "<rect width='14' height='14' fill='#f0f8ff'/>",
            "<path d='M0 0 L14 14 M14 0 L0 14' stroke='#aaccff' stroke-width='0.6'/>",
            "</pattern>",
            "<!-- Material hatching patterns (architectural standards) -->",
            "<pattern id='concrete' width='16' height='16' patternUnits='userSpaceOnUse'>",
            "<rect width='16' height='16' fill='#f5f5f5'/>",
            "<circle cx='3' cy='3' r='1' fill='#999'/>",
            "<circle cx='11' cy='7' r='1.2' fill='#888'/>",
            "<circle cx='7' cy='13' r='0.8' fill='#aaa'/>",
            "<circle cx='14' cy='11' r='1' fill='#999'/>",
            "</pattern>",
            "<pattern id='masonry' width='20' height='10' patternUnits='userSpaceOnUse'>",
            "<rect width='20' height='10' fill='#f8f8f8'/>",
            "<rect x='0' y='0' width='10' height='5' fill='none' stroke='#666' stroke-width='0.5'/>",
            "<rect x='10' y='0' width='10' height='5' fill='none' stroke='#666' stroke-width='0.5'/>",
            "<rect x='5' y='5' width='10' height='5' fill='none' stroke='#666' stroke-width='0.5'/>",
            "</pattern>",
            "<pattern id='wood-grain' width='24' height='8' patternUnits='userSpaceOnUse'>",
            "<rect width='24' height='8' fill='#fdf6e3'/>",
            "<path d='M0 2 Q6 1 12 2 T24 2' stroke='#d4a574' stroke-width='0.4' fill='none'/>",
            "<path d='M0 4 Q8 3.5 16 4 T24 4' stroke='#c5956a' stroke-width='0.3' fill='none'/>",
            "<path d='M0 6 Q10 5.5 20 6 T24 6' stroke='#d4a574' stroke-width='0.4' fill='none'/>",
            "</pattern>",
            "<pattern id='steel' width='12' height='12' patternUnits='userSpaceOnUse'>",
            "<rect width='12' height='12' fill='#e8e8e8'/>",
            "<path d='M0 0 L12 12 M0 12 L12 0' stroke='#666' stroke-width='0.6'/>",
            "</pattern>",
            "<pattern id='insulation' width='18' height='12' patternUnits='userSpaceOnUse'>",
            "<rect width='18' height='12' fill='#fff8dc'/>",
            "<path d='M0 6 Q4.5 2 9 6 T18 6' stroke='#daa520' stroke-width='1' fill='none'/>",
            "<path d='M0 6 Q4.5 10 9 6 T18 6' stroke='#daa520' stroke-width='1' fill='none'/>",
            "</pattern>",
            "<pattern id='earth' width='16' height='16' patternUnits='userSpaceOnUse'>",
            "<rect width='16' height='16' fill='#f4f0e6'/>",
            "<circle cx='2' cy='3' r='0.8' fill='#8b7355'/>",
            "<circle cx='8' cy='1' r='1.2' fill='#a0825a'/>",
            "<circle cx='13' cy='5' r='0.6' fill='#9d8668'/>",
            "<circle cx='5' cy='8' r='1' fill='#8b7355'/>",
            "<circle cx='11' cy='11' r='0.9' fill='#a0825a'/>",
            "<circle cx='3' cy='14' r='0.7' fill='#9d8668'/>",
            "<circle cx='15' cy='13' r='1.1' fill='#8b7355'/>",
            "</pattern>",
            "</defs>",
            f"<rect width='{width}' height='{height}' fill='white'/>",
            "<g id='spaces'>",
        ]

        # Simple fit: compute bounds of all points (walls and spaces)
        pts: List[Tuple[float, float]] = []
        for w in self.walls:
            pts.extend([w.start, w.end])
        for s in self.spaces:
            pts.extend(list(s.boundary))
        if not pts:
            svg.append("</svg>")
            return "".join(svg)
        minx = min(p[0] for p in pts)
        miny = min(p[1] for p in pts)
        maxx = max(p[0] for p in pts)
        maxy = max(p[1] for p in pts)
        w = max(1.0, maxx - minx)
        h = max(1.0, maxy - miny)
        margin = 80.0
        sx = (width - margin * 2) / w
        sy = (height - margin * 2) / h
        s = min(sx, sy)
        tx = margin - minx * s
        ty = margin + maxy * s

        def T(x: float, y: float) -> Tuple[float, float]:
            return (x * s + tx, -y * s + ty)

        # Snap a point to 1-inch grid in world units (mm)
        def snap_mm(x: float, y: float, grid_in: float = 1.0) -> Tuple[float, float]:
            grid_mm = grid_in * 25.4
            return (
                round(x / grid_mm) * grid_mm,
                round(y / grid_mm) * grid_mm,
            )

        def draw_ticks(x: float, y: float, angle_deg: float, length_px: float = 8.0) -> str:
            ang = math.radians(angle_deg + 45.0)
            dx = math.cos(ang) * (length_px / 2.0)
            dy = math.sin(ang) * (length_px / 2.0)
            return f"<line x1='{x-dx:.1f}' y1='{y-dy:.1f}' x2='{x+dx:.1f}' y2='{y+dy:.1f}' class='tick'/>"

        def dim_with_ticks(x1: float, y1: float, x2: float, y2: float, label: str, offset: float = 22.0, ext: float = 10.0) -> List[str]:
            out: List[str] = []
            dx = x2 - x1
            dy = y2 - y1
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            px, py = -uy, ux
            # extension points
            ex1x, ex1y = x1 + px * (offset + ext), y1 + py * (offset + ext)
            ex2x, ex2y = x2 + px * (offset + ext), y2 + py * (offset + ext)
            ox1, oy1 = x1 + px * offset, y1 + py * offset
            ox2, oy2 = x2 + px * offset, y2 + py * offset
            # extension lines
            out.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{ex1x:.1f}' y2='{ex1y:.1f}' class='dim'/>")
            out.append(f"<line x1='{x2:.1f}' y1='{y2:.1f}' x2='{ex2x:.1f}' y2='{ex2y:.1f}' class='dim'/>")
            # dimension line
            out.append(f"<line x1='{ox1:.1f}' y1='{oy1:.1f}' x2='{ox2:.1f}' y2='{oy2:.1f}' class='dim'/>")
            angle = math.degrees(math.atan2(oy2 - oy1, ox2 - ox1))
            # ticks
            out.append(draw_ticks(ox1, oy1, angle, 8.0))
            out.append(draw_ticks(ox2, oy2, angle, 8.0))
            # label
            midx, midy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
            out.append(f"<text x='{midx:.1f}' y='{midy-4:.1f}' class='dimtxt' text-anchor='middle'>{label}</text>")
            return out

        # Unit helpers (assume input units are mm)
        def mm_to_feet_inches_str(mm: float) -> str:
            total_inches = mm / 25.4
            feet = int(total_inches // 12)
            inches = total_inches - feet * 12
            inches_rounded = int(round(inches))
            if inches_rounded == 12:
                feet += 1
                inches_rounded = 0
            return f"{feet}'-{inches_rounded}\""

        def mm2_to_sqft(mm2: float) -> float:
            return mm2 / 92903.04

        # Render spaces (fills)
        for sp in self.spaces:
            if len(sp.boundary) >= 3:
                pts_t = [T(x, y) for x, y in sp.boundary]
                d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts_t) + " Z"
                name_lower = (sp.name or "").lower()
                space_type = (sp.type or "").lower()
                
                # Apply appropriate material patterns based on room function
                if any(k in name_lower for k in ["bath", "toilet", "wash", "wc", "powder"]):
                    # Wet areas - ceramic tile
                    svg.append(f"<path d='{d}' fill='url(#tile)' stroke='none'/>")
                elif any(k in name_lower for k in ["kitchen", "utility", "laundry"]):
                    # Service areas - concrete floors
                    svg.append(f"<path d='{d}' fill='url(#concrete)' stroke='none' opacity='0.3'/>")
                elif any(k in name_lower for k in ["garage", "mechanical", "storage"]):
                    # Utilitarian spaces - concrete
                    svg.append(f"<path d='{d}' fill='url(#concrete)' stroke='none' opacity='0.4'/>")
                elif any(k in name_lower for k in ["deck", "porch", "patio"]):
                    # Outdoor spaces - wood decking
                    svg.append(f"<path d='{d}' fill='url(#wood-grain)' stroke='none' opacity='0.3'/>")
                elif any(k in name_lower for k in ["stair", "steps"]):
                    # Stairs - typically concrete or wood
                    svg.append(f"<path d='{d}' fill='url(#concrete)' stroke='none' opacity='0.2'/>")
                elif any(k in name_lower for k in ["basement", "cellar", "foundation"]):
                    # Below grade - concrete
                    svg.append(f"<path d='{d}' fill='url(#concrete)' stroke='none' opacity='0.5'/>")
                else:
                    # Standard living spaces - light fill
                    svg.append(f"<path d='{d}' class='roomfill' stroke='none'/>")
        svg.append("</g>")
        svg.append("<g id='walls'>")

        # Render walls with openings cut out
        for wll in self.walls:
            # collect openings on this wall
            kids = [op for op in self.openings if op.wall_id == wll.id]
            # base wall strip in world coords, then transform after cutting
            # Split the wall line into spans excluding openings
            ax, ay = snap_mm(*wll.start)
            bx, by = snap_mm(*wll.end)
            spans: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [((ax, ay), (bx, by))]
            for op in kids:
                (ox1, oy1), (ox2, oy2) = self._opening_span(wll, op)
                ox1, oy1 = snap_mm(ox1, oy1)
                ox2, oy2 = snap_mm(ox2, oy2)
                new_spans: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
                for (sx1, sy1), (sx2, sy2) in spans:
                    # project onto wall direction
                    ux, uy, L = self._line_dir((sx1, sy1), (sx2, sy2))
                    if L == 0:
                        continue
                    # coordinates along span
                    def proj(t: Tuple[float, float]) -> float:
                        return (t[0] - sx1) * ux + (t[1] - sy1) * uy
                    a0, a1 = 0.0, L
                    oc0 = max(0.0, min(L, proj((ox1, oy1))))
                    oc1 = max(0.0, min(L, proj((ox2, oy2))))
                    lo, hi = sorted((oc0, oc1))
                    pad = max(5.0, wll.thickness * 0.05)
                    # left segment
                    if lo - pad > a0:
                        na = (sx1 + ux * (a0), sy1 + uy * (a0))
                        nb = (sx1 + ux * (lo - pad), sy1 + uy * (lo - pad))
                        new_spans.append((na, nb))
                    # right segment
                    if hi + pad < a1:
                        na = (sx1 + ux * (hi + pad), sy1 + uy * (hi + pad))
                        nb = (sx1 + ux * (a1), sy1 + uy * (a1))
                        new_spans.append((na, nb))
                spans = new_spans or spans
            # draw spans as thick strips
            # Use visual thickness constants (px)
            thickness = self.outer_wall_px if wll.type == "exterior" else self.inner_wall_px
            for (sx1, sy1), (sx2, sy2) in spans:
                X1, Y1 = T(sx1, sy1)
                X2, Y2 = T(sx2, sy2)
                d = self._wall_strip((X1, Y1), (X2, Y2), thickness)
                if d:
                    # Professional line weight assignment based on wall type with material patterns
                    if getattr(wll, "subtype", None) == "plumbing":
                        klass = "wall-plumbing"
                        fill_attr = ""  # Keep existing plumbing wall appearance
                    elif wll.type == "exterior":
                        klass = "wall-exterior"
                        fill_attr = " fill='url(#masonry)'"  # Exterior walls typically masonry/concrete
                    elif wll.type == "interior":
                        klass = "wall-interior" 
                        # Check wall thickness to determine material
                        if wll.thickness > 150:  # Thick interior walls = masonry/concrete
                            fill_attr = " fill='url(#concrete)'"
                        else:  # Thin interior walls = wood frame/drywall
                            fill_attr = " fill='url(#wood-grain)'"
                    else:
                        klass = "wall"  # default fallback
                        fill_attr = ""
                    svg.append(f"<path d='{d}' class='{klass}'{fill_attr}/>")
        svg.append("</g>")
        # Structural elements (columns/beams)
        svg.append("<g id='structural'>")
        self._add_structural_elements(svg, T)
        svg.append("</g>")
        svg.append("<g id='openings'>")

        # Render openings symbols
        for op in self.openings:
            wll = next((w for w in self.walls if w.id == op.wall_id), None)
            if not wll:
                continue
            (ox1, oy1), (ox2, oy2) = self._opening_span(wll, op)
            X1, Y1 = T(ox1, oy1)
            X2, Y2 = T(ox2, oy2)
            angle = math.degrees(math.atan2(Y2 - Y1, X2 - X1))
            length = math.hypot(X2 - X1, Y2 - Y1)
            # Door/window height uses same visual wall thickness
            thickness_px = self.outer_wall_px if wll.type == "exterior" else self.inner_wall_px
            if op.type == "door":
                # Professional door symbol honoring handing (L/R) and swing (in/out)
                door_thickness = 2.0  # Door leaf thickness
                frame_thickness = 1.5  # Frame thickness
                h = self._clamp(0.9 * thickness_px, 6.0, 12.0)
                handing = str(op.metadata.get("handing", "RHR")).upper()
                swing_dir = str(op.metadata.get("swing", "in")).lower()
                hinge_at_end = handing.startswith("R") and not handing.endswith("F")
                svg.append(f"<g transform='translate({X1:.1f},{Y1:.1f}) rotate({angle:.1f})'>")
                inner_tx = length if hinge_at_end else 0.0
                inner_sx = -1 if hinge_at_end else 1
                svg.append(f"<g transform='translate({inner_tx:.1f},0) scale({inner_sx},1)'>")
                # Door frame (slightly wider than opening)
                frame_extend = 2.0
                svg.append(f"<rect x='-{frame_extend:.1f}' y='-{h/2+frame_thickness:.1f}' width='{length+2*frame_extend:.1f}' height='{h+2*frame_thickness:.1f}' class='door' fill='#D2691E' opacity='0.3'/>")
                
                # Door opening (clear span)
                svg.append(f"<rect x='0' y='-{h/2:.1f}' width='{length:.1f}' height='{h:.1f}' fill='white' stroke='none'/>")
                
                # Door leaf (typically 50mm thick in plan)
                leaf_h = door_thickness
                svg.append(f"<rect x='2' y='-{leaf_h/2:.1f}' width='{length-4:.1f}' height='{leaf_h:.1f}' class='door'/>")
                
                # Door handle/knob (at 1/8 point from hinge)
                handle_x = length * 0.85
                svg.append(f"<circle cx='{handle_x:.1f}' cy='0' r='1.5' class='door'/>")
                
                # Swing arc (90 degree) with direction
                swing_radius = length - 4
                sy = 1.0 if swing_dir == "in" else -1.0
                svg.append(f"<path d='M 2 0 A {swing_radius:.1f} {swing_radius:.1f} 0 0 1 {2+swing_radius*0.7:.1f} {sy * -swing_radius*0.7:.1f}' class='door-swing'/>")
                
                # Hinge side marker
                svg.append(f"<line x1='2' y1='-{h/2:.1f}' x2='2' y2='{h/2:.1f}' stroke='#8B4513' stroke-width='2'/>")
                svg.append("</g>")
                
            else:  # window
                # Professional window symbol with operation (fixed/casement/awning/slider/double_hung)
                sill_depth = 4.0
                frame_width = 1.5
                h = self._clamp(0.9 * thickness_px, 8.0, 16.0)
                operation = str(op.metadata.get("window_operation", "fixed"))
                svg.append(f"<g transform='translate({X1:.1f},{Y1:.1f}) rotate({angle:.1f})'>")
                
                # Window sill (projects beyond wall)
                sill_extend = 3.0
                svg.append(f"<rect x='-{sill_extend:.1f}' y='-{h/2+sill_depth:.1f}' width='{length+2*sill_extend:.1f}' height='{sill_depth:.1f}' class='window-sill'/>")
                
                # Window frame
                svg.append(f"<rect x='0' y='-{h/2:.1f}' width='{length:.1f}' height='{h:.1f}' class='window'/>")
                
                # Operation graphics
                if operation.startswith("casement"):
                    left_hinge = "left" in operation
                    hx = 0 if left_hinge else length
                    x2 = length * (0.7 if left_hinge else 0.3)
                    svg.append(f"<line x1='{hx:.1f}' y1='0' x2='{x2:.1f}' y2='-{h/2 - 1:.1f}' class='window' stroke-dasharray='2,1'/>")
                elif operation.startswith("awning"):
                    svg.append(f"<path d='M 0 {-h/2:.1f} L {length/2:.1f} {-h/2+6:.1f} L {length:.1f} {-h/2:.1f}' class='window' stroke-dasharray='2,1' fill='none'/>")
                elif operation.startswith("slider"):
                    svg.append(f"<rect x='{length*0.05:.1f}' y='-{h/2-1:.1f}' width='{length*0.4:.1f}' height='{h-2:.1f}' fill='none' stroke='#0066CC' stroke-width='0.8'/>")
                    svg.append(f"<rect x='{length*0.55:.1f}' y='-{h/2-1:.1f}' width='{length*0.4:.1f}' height='{h-2:.1f}' fill='none' stroke='#0066CC' stroke-width='0.8'/>")
                elif operation == "double_hung":
                    svg.append(f"<line x1='{length/2:.1f}' y1='-{h/2-2:.1f}' x2='{length/2:.1f}' y2='{-2:.1f}' class='window' stroke-dasharray='2,1'/>")
                    svg.append(f"<line x1='{length/2:.1f}' y1='{2:.1f}' x2='{length/2:.1f}' y2='{h/2-2:.1f}' class='window' stroke-dasharray='2,1'/>")
                
                # Window opening direction indicator (for operable windows)
                if length > 50:  # Only for larger windows
                    vent_x = length * 0.75
                    svg.append(f"<line x1='{vent_x:.1f}' y1='-{h/4:.1f}' x2='{vent_x+8:.1f}' y2='-{h/4:.1f}' stroke='#0066CC' stroke-width='0.6' stroke-dasharray='2,1'/>")
                
                svg.append("</g>")
                
                # Add opening dimensions
                self._add_opening_dimensions(svg, op, wll, X1, Y1, X2, Y2, length, T)

        # Wall endcaps at openings (simple perpendicular caps for visual polish)
        cap_len = 6.0
        for op in self.openings:
            wll = next((w for w in self.walls if w.id == op.wall_id), None)
            if not wll:
                continue
            (ox1, oy1), (ox2, oy2) = self._opening_span(wll, op)
            X1, Y1 = T(*snap_mm(ox1, oy1))
            X2, Y2 = T(*snap_mm(ox2, oy2))
            # perpendicular vector
            dx, dy = X2 - X1, Y2 - Y1
            L = math.hypot(dx, dy) or 1.0
            px, py = -dy / L, dx / L
            svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X1+px*cap_len:.1f}' y2='{Y1+py*cap_len:.1f}' class='cap'/>")
            svg.append(f"<line x1='{X2:.1f}' y1='{Y2:.1f}' x2='{X2+px*cap_len:.1f}' y2='{Y2+py*cap_len:.1f}' class='cap'/>")

        # Centerline for plumbing walls
        for wll in self.walls:
            if getattr(wll, "subtype", None) != "plumbing":
                continue
            X1, Y1 = T(*snap_mm(*wll.start))
            X2, Y2 = T(*snap_mm(*wll.end))
            svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' class='pl-center'/>")
        svg.append("</g>")
        
        # Grid system layer (behind everything except spaces)
        svg.append("<g id='grid'>")
        self._add_architectural_grid(svg, T, width, height)
        svg.append("</g>")
        
        # Fixtures layer
        svg.append("<g id='fixtures'>")
        for sp in self.spaces:
            self._add_fixture_symbols(svg, sp, T)
        svg.append("</g>")
        
        # Sheet-mode overlays
        if self.sheet_mode == "rcp":
            svg.append("<g id='rcp-overlays'>")
            self._add_rcp_overlays(svg, T)
            svg.append("</g>")
        elif self.sheet_mode == "power":
            svg.append("<g id='power-circuits'>")
            self._add_power_overlays(svg, T)
            svg.append("</g>")
        elif self.sheet_mode == "plumbing":
            svg.append("<g id='plumbing-runs'>")
            self._add_plumbing_overlays(svg, T)
            svg.append("</g>")

        # Code compliance layer
        svg.append("<g id='code-compliance'>")
        self._add_code_compliance_indicators(svg, T)
        svg.append("</g>")
        
        svg.append("<g id='annotations'>")

        # Labels
        for sp in self.spaces:
            if not sp.boundary:
                continue
            cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
            cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
            CX, CY = T(cx, cy)
            # area in sq ft (input units mm)
            def poly_area(pts: List[Tuple[float, float]]) -> float:
                a = 0.0
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    a += x1 * y2 - x2 * y1
                return abs(a) * 0.5
            area_mm2 = poly_area(sp.boundary)
            area_sqft = mm2_to_sqft(area_mm2)
            # Enhanced technical annotations
            self._add_technical_room_annotations(svg, sp, CX, CY, area_sqft, T)
            # Stairwell detail
            if sp.name and sp.name.lower().startswith("stair"):
                # draw stair treads across the short dimension
                xs = [p[0] for p in sp.boundary]
                ys = [p[1] for p in sp.boundary]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                w_mm = maxx - minx
                h_mm = maxy - miny
                # choose direction: vertical if taller, else horizontal
                treads = 7
                if h_mm >= w_mm:
                    # vertical stairs: draw horizontal treads
                    for i in range(1, treads):
                        yy = miny + (i * h_mm / treads)
                        X1, Y1 = T(minx + 10, yy)
                        X2, Y2 = T(maxx - 10, yy)
                        svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#333' stroke-width='1'/>")
                    # arrow up
                    AX, AY = T(minx + w_mm * 0.5, miny + h_mm * 0.2)
                    svg.append(f"<path d='M {AX:.1f} {AY:.1f} l 8 12 l -16 0 Z' fill='#333'/>")
                else:
                    # horizontal stairs: draw vertical treads
                    for i in range(1, treads):
                        xx = minx + (i * w_mm / treads)
                        X1, Y1 = T(xx, miny + 10)
                        X2, Y2 = T(xx, maxy - 10)
                        svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#333' stroke-width='1'/>")
                    AX, AY = T(minx + w_mm * 0.2, miny + h_mm * 0.5)
                    svg.append(f"<path d='M {AX:.1f} {AY:.1f} l 12 -8 l 0 16 Z' fill='#333'/>")
            # simple dims for bounding box
            xs = [p[0] for p in sp.boundary]
            ys = [p[1] for p in sp.boundary]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            X1, Y1 = T(minx, miny)
            X2, Y2 = T(maxx, miny)
            X3, Y3 = T(maxx, maxy)
            # dimension lines with small offsets
            off = 14.0
            for part in dim_with_ticks(X1, Y1, X2, Y2, mm_to_feet_inches_str(maxx-minx), off):
                svg.append(part)
            for part in dim_with_ticks(X2, Y2, X3, Y3, mm_to_feet_inches_str(maxy-miny), off):
                svg.append(part)

        # Overall outer dimensions (based on all geometry pts)
        TX1, TY1 = T(minx, miny)
        TX2, TY2 = T(maxx, miny)
        # extension lines
        svg.append(f"<line x1='{TX1:.1f}' y1='{TY1:.1f}' x2='{TX1:.1f}' y2='{TY1-36:.1f}' class='dim'/>")
        svg.append(f"<line x1='{TX2:.1f}' y1='{TY2:.1f}' x2='{TX2:.1f}' y2='{TY2-36:.1f}' class='dim'/>")
        for part in dim_with_ticks(TX1, TY1, TX2, TY2, f"{mm_to_feet_inches_str(maxx-minx)} overall", 28.0):
            svg.append(part)
        RX1, RY1 = T(maxx, miny)
        RX2, RY2 = T(maxx, maxy)
        svg.append(f"<line x1='{RX1:.1f}' y1='{RY1:.1f}' x2='{RX1+36:.1f}' y2='{RY1:.1f}' class='dim'/>")
        svg.append(f"<line x1='{RX2:.1f}' y1='{RY2:.1f}' x2='{RX2+36:.1f}' y2='{RY2:.1f}' class='dim'/>")
        for part in dim_with_ticks(RX1, RY1, RX2, RY2, f"{mm_to_feet_inches_str(maxy-miny)} overall", 28.0):
            svg.append(part)
            
        # ADVANCED DIMENSIONING SYSTEM
        self._add_chained_dimensions(svg, T, width, height, margin)
        self._add_elevation_markers(svg, T, width, height, margin)
        self._add_radial_dimensions(svg, T)
        
        # North arrow and scale bar
        NAx, NAy = width - 120, 120
        na_size = 50
        svg.append(
            f"<g transform='translate({NAx},{NAy})'>"
            f"<polygon points='0,-{na_size} {na_size/2},0 0,{na_size/4} -{na_size/2},0' fill='#FF0000' stroke='#000' stroke-width='2'/>"
            f"<text x='0' y='{na_size+18}' text-anchor='middle' class='label'>NORTH</text>"
            "</g>"
        )
        SBx, SBy = width - 220, height - 100
        svg.append(
            f"<g transform='translate({SBx},{SBy})'>"
            "<rect x='0' y='0' width='120' height='12' fill='none' stroke='#000' stroke-width='2'/>"
            "<line x1='0' y1='0' x2='0' y2='16' stroke='#000' stroke-width='2'/>"
            "<line x1='30' y1='0' x2='30' y2='16' stroke='#000' stroke-width='2'/>"
            "<line x1='60' y1='0' x2='60' y2='16' stroke='#000' stroke-width='2'/>"
            "<line x1='90' y1='0' x2='90' y2='16' stroke='#000' stroke-width='2'/>"
            "<line x1='120' y1='0' x2='120' y2='16' stroke='#000' stroke-width='2'/>"
            "<text x='60' y='34' text-anchor='middle' class='label'>0 10 20 30 40 ft</text>"
            "</g>"
        )
        svg.append("</g>")
        # Title block
        svg.append(f"<rect x='{width-260:.1f}' y='{height-120:.1f}' width='240' height='110' fill='white' stroke='#000' stroke-width='1' />")
        title_map = {
            "floor": "ARCHITECTURAL FLOOR PLAN",
            "rcp": "REFLECTED CEILING PLAN",
            "power": "POWER/LIGHTING PLAN",
            "plumbing": "PLUMBING PLAN",
        }
        sheet_title = title_map.get(getattr(self, "sheet_mode", "floor"), "ARCHITECTURAL FLOOR PLAN")
        svg.append(f"<text x='{width-140:.1f}' y='{height-100:.1f}' class='title' text-anchor='middle'>{sheet_title}</text>")
        svg.append(f"<text x='{width-140:.1f}' y='{height-80:.1f}' class='sub' text-anchor='middle'>Scale: 1:100</text>")
        svg.append(f"<text x='{width-140:.1f}' y='{height-62:.1f}' class='sub' text-anchor='middle'>Project: HouseBrain</text>")
        total_area_sqft = 0.0
        for sp in self.spaces:
            if len(sp.boundary) >= 3:
                # area in m2
                def A(pts: List[Tuple[float, float]]):
                    a = 0.0
                    for i in range(len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[(i + 1) % len(pts)]
                        a += x1 * y2 - x2 * y1
                    return abs(a) * 0.5
                total_area_sqft += mm2_to_sqft(A(sp.boundary))
        svg.append(f"<text x='{width-140:.1f}' y='{height-44:.1f}' class='sub' text-anchor='middle'>Total area: {total_area_sqft:.0f} sq ft</text>")
        svg.append(f"<text x='{width-140:.1f}' y='{height-26:.1f}' class='sub' text-anchor='middle'>Drawn by: HB-AI</text>")

        svg.append("</svg>")
        return "".join(svg)

    def _segments_intersect(self, a1, a2, b1, b2):
        """Return intersection point of segments a1-a2 and b1-b2 if exists, else None."""
        (x1, y1), (x2, y2) = a1, a2
        (x3, y3), (x4, y4) = b1, b2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 * x2 - y1 * x2) * 0.0)  # dummy to keep structure
        # Proper intersection computation
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        # Check within segment bounds with tolerance
        def within(a, b, p):
            return min(a, b) - 1e-6 <= p <= max(a, b) + 1e-6
        if within(x1, x2, px) and within(y1, y2, py) and within(x3, x4, px) and within(y3, y4, py):
            return (px, py)
        return None

    def _add_structural_elements(self, svg: List[str], T) -> None:
        """Detect wall intersections and place columns; draw simple beams between columns."""
        # Gather intersections
        points: List[tuple] = []
        for i, w1 in enumerate(self.walls):
            for j, w2 in enumerate(self.walls):
                if j <= i:
                    continue
                p = self._segments_intersect(w1.start, w1.end, w2.start, w2.end)
                if p is not None:
                    points.append(p)

        # Place columns at intersections
        # Column size depends on wall types; default 300mm
        col_mm = 300.0
        for (px, py) in points:
            CX, CY = T(px, py)
            size_px = max(8.0, col_mm * self.scale / 100.0)
            svg.append(
                f"<rect x='{CX - size_px/2:.1f}' y='{CY - size_px/2:.1f}' width='{size_px:.1f}' height='{size_px:.1f}' fill='url(#concrete)' stroke='#333' stroke-width='1'/>"
            )

        # Optional: draw beams between nearest columns (simple visual ties)
        # Connect points that are roughly aligned horizontally/vertically
        aligned_pairs: List[tuple] = []
        for i in range(len(points)):
            x1, y1 = points[i]
            for j in range(i + 1, len(points)):
                x2, y2 = points[j]
                if abs(y1 - y2) < 1e-3 or abs(x1 - x2) < 1e-3:
                    aligned_pairs.append(((x1, y1), (x2, y2)))
        for (a, b) in aligned_pairs:
            X1, Y1 = T(*a)
            X2, Y2 = T(*b)
            svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#555' stroke-width='2' stroke-dasharray='6,3' />")

    def _add_rcp_overlays(self, svg: List[str], T) -> None:
        """Add reflected ceiling plan overlays: ceiling grid and RCP fixtures."""
        for sp in self.spaces:
            if not sp.boundary:
                continue
            xs = [p[0] for p in sp.boundary]
            ys = [p[1] for p in sp.boundary]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            # 600x600 ceiling grid
            spacing = 600.0
            x = minx - ((minx) % spacing)
            while x <= maxx:
                X1, Y1 = T(x, miny)
                X2, Y2 = T(x, maxy)
                svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#bbb' stroke-width='0.6' stroke-dasharray='6,4' />")
                x += spacing
            y = miny - ((miny) % spacing)
            while y <= maxy:
                X1, Y1 = T(minx, y)
                X2, Y2 = T(maxx, y)
                svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#bbb' stroke-width='0.6' stroke-dasharray='6,4' />")
                y += spacing
            # Central ceiling fixture marker
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            CX, CY = T(cx, cy)
            svg.append(f"<circle cx='{CX:.1f}' cy='{CY:.1f}' r='8' fill='none' stroke='#222' stroke-width='1.2' />")
            svg.append(f"<text x='{CX:.1f}' y='{CY+4:.1f}' class='sub' text-anchor='middle' font-size='7'>CLG</text>")

    def _add_power_overlays(self, svg: List[str], T) -> None:
        """Add power circuits and panel location overlays."""
        # Choose a panel location near first space min corner
        if not self.spaces:
            return
        sp0 = self.spaces[0]
        xs = [p[0] for p in sp0.boundary] if sp0.boundary else [0, 0]
        ys = [p[1] for p in sp0.boundary] if sp0.boundary else [0, 0]
        panel_x, panel_y = min(xs) + 300, min(ys) + 300
        PX, PY = T(panel_x, panel_y)
        svg.append(f"<rect x='{PX-12:.1f}' y='{PY-12:.1f}' width='24' height='24' fill='#333' stroke='white' stroke-width='1' />")
        svg.append(f"<text x='{PX:.1f}' y='{PY+4:.1f}' class='sub' text-anchor='middle' font-size='7' fill='white'>PNL</text>")
        # Draw notional circuits to room centers
        for sp in self.spaces:
            if not sp.boundary:
                continue
            cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
            cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
            CX, CY = T(cx, cy)
            svg.append(f"<path d='M {PX:.1f} {PY:.1f} Q {(PX+CX)/2:.1f} {(PY):.1f} {CX:.1f} {CY:.1f}' stroke='#AA00FF' stroke-width='1.4' fill='none' stroke-dasharray='10,4' />")
        # Legend
        lx, ly = PX + 40, PY + 20
        svg.append(f"<rect x='{lx-8:.1f}' y='{ly-20:.1f}' width='130' height='40' fill='white' stroke='#AA00FF' stroke-width='1' opacity='0.9' />")
        svg.append(f"<text x='{lx:.1f}' y='{ly-6:.1f}' class='sub' font-size='8' fill='#AA00FF'>Panel (PNL)</text>")
        svg.append(f"<text x='{lx:.1f}' y='{ly+8:.1f}' class='sub' font-size='8' fill='#AA00FF'>Dashed: Circuit run</text>")

    def _add_plumbing_overlays(self, svg: List[str], T) -> None:
        """Add plumbing supply (blue/red) and drain (brown) schematic runs."""
        # Main enters at lower-left
        if not self.spaces:
            return
        all_pts = []
        for sp in self.spaces:
            all_pts.extend(sp.boundary)
        if not all_pts:
            return
        minx = min(p[0] for p in all_pts)
        miny = min(p[1] for p in all_pts)
        maxx = max(p[0] for p in all_pts)
        # Main supply (blue) along bottom wall
        X1, Y1 = T(minx + 100, miny + 150)
        X2, Y2 = T(maxx - 100, miny + 150)
        svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#007BFF' stroke-width='2' />")
        # Branches to wet rooms
        for sp in self.spaces:
            name_lower = (sp.name or '').lower()
            if any(k in name_lower for k in ["bath", "toilet", "wash", "wc", "kitchen", "utility", "laundry"]):
                cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
                cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
                CX, CY = T(cx, cy)
                svg.append(f"<line x1='{CX:.1f}' y1='{CY:.1f}' x2='{CX:.1f}' y2='{Y1:.1f}' stroke='#007BFF' stroke-width='1.5' stroke-dasharray='6,3' />")
                # Hot water (red) short branch
                svg.append(f"<line x1='{CX+8:.1f}' y1='{CY:.1f}' x2='{CX+8:.1f}' y2='{Y1:.1f}' stroke='#FF4444' stroke-width='1.2' stroke-dasharray='4,3' />")
        # Drain (brown) parallel to supply
        Yd = (Y1 + 20)
        svg.append(f"<line x1='{X1:.1f}' y1='{Yd:.1f}' x2='{X2:.1f}' y2='{Yd:.1f}' stroke='#8B4513' stroke-width='2' stroke-dasharray='10,6' />")

    def _add_technical_room_annotations(self, svg: List[str], sp: Space, CX: float, CY: float, area_sqft: float, T) -> None:
        """Add comprehensive technical annotations for each room."""
        name_lower = (sp.name or "").lower()
        
        # Room name (main label)
        svg.append(f"<text x='{CX:.1f}' y='{CY-20:.1f}' class='label' text-anchor='middle' font-weight='bold'>{sp.name}</text>")
        
        # Area
        svg.append(f"<text x='{CX:.1f}' y='{CY-6:.1f}' class='sub' text-anchor='middle'>{area_sqft:.0f} sq ft</text>")
        
        # Ceiling height (based on room type)
        ceiling_height = self._get_ceiling_height(name_lower)
        svg.append(f"<text x='{CX:.1f}' y='{CY+8:.1f}' class='sub' text-anchor='middle' font-size='8'>CLG HT: {ceiling_height}</text>")
        
        # Floor finish (based on room type)
        floor_finish = self._get_floor_finish(name_lower)
        svg.append(f"<text x='{CX:.1f}' y='{CY+20:.1f}' class='sub' text-anchor='middle' font-size='8'>FLOOR: {floor_finish}</text>")
        
        # Special annotations for specific room types
        if any(k in name_lower for k in ["bath", "toilet", "wash", "wc"]):
            svg.append(f"<text x='{CX:.1f}' y='{CY+32:.1f}' class='sub' text-anchor='middle' font-size='7'>WATERPROOF</text>")
        elif any(k in name_lower for k in ["kitchen"]):
            svg.append(f"<text x='{CX:.1f}' y='{CY+32:.1f}' class='sub' text-anchor='middle' font-size='7'>VENTILATION REQ'D</text>")
        elif any(k in name_lower for k in ["bed", "master"]):
            svg.append(f"<text x='{CX:.1f}' y='{CY+32:.1f}' class='sub' text-anchor='middle' font-size='7'>EGRESS WINDOW</text>")

    def _get_ceiling_height(self, name_lower: str) -> str:
        """Return appropriate ceiling height based on room type."""
        if any(k in name_lower for k in ["bath", "toilet", "utility", "laundry"]):
            return "8'-0\""  # Standard service rooms
        elif any(k in name_lower for k in ["living", "dining", "family", "great"]):
            return "9'-0\""  # Higher ceilings for main living areas
        elif any(k in name_lower for k in ["master", "bed"]):
            return "9'-0\""  # Standard bedroom height
        elif any(k in name_lower for k in ["kitchen"]):
            return "8'-6\""  # Kitchen with soffits
        elif any(k in name_lower for k in ["basement", "cellar"]):
            return "7'-6\""  # Lower basement ceilings
        elif any(k in name_lower for k in ["garage"]):
            return "8'-0\""  # Garage height
        else:
            return "8'-6\""  # Default height

    def _get_floor_finish(self, name_lower: str) -> str:
        """Return appropriate floor finish based on room type."""
        if any(k in name_lower for k in ["bath", "toilet", "wash", "wc"]):
            return "CERAMIC TILE"
        elif any(k in name_lower for k in ["kitchen"]):
            return "CERAMIC TILE"
        elif any(k in name_lower for k in ["utility", "laundry"]):
            return "VINYL TILE"
        elif any(k in name_lower for k in ["living", "dining", "family", "bed", "master"]):
            return "HARDWOOD"
        elif any(k in name_lower for k in ["entry", "entrance", "foyer"]):
            return "CERAMIC TILE"
        elif any(k in name_lower for k in ["garage"]):
            return "CONCRETE"
        elif any(k in name_lower for k in ["basement", "cellar"]):
            return "CONCRETE"
        elif any(k in name_lower for k in ["deck", "porch", "patio"]):
            return "COMPOSITE DECK"
        else:
            return "CARPET"  # Default for unspecified rooms

    def _add_opening_dimensions(self, svg: List[str], op: Opening, wll: Wall, X1: float, Y1: float, X2: float, Y2: float, length: float, T) -> None:
        """Add dimension annotations for doors and windows."""
        # Convert opening width from pixels back to mm, then to feet-inches
        width_mm = op.width
        width_ft_in = self._mm_to_feet_inches_str(width_mm)
        
        # Position dimension text above/below the opening
        angle = math.degrees(math.atan2(Y2 - Y1, X2 - X1))
        midX, midY = (X1 + X2) / 2, (Y1 + Y2) / 2
        
        # Offset dimension text perpendicular to opening
        offset_distance = 25.0
        angle_rad = math.radians(angle)
        perp_x = -math.sin(angle_rad) * offset_distance
        perp_y = math.cos(angle_rad) * offset_distance
        
        dim_x = midX + perp_x
        dim_y = midY + perp_y
        
        # Opening type abbreviation
        op_type = "DR" if op.type == "door" else "WN"
        
        # Add dimension with background for readability
        svg.append(f"<circle cx='{dim_x:.1f}' cy='{dim_y:.1f}' r='12' fill='white' stroke='#333' stroke-width='0.8'/>")
        svg.append(f"<text x='{dim_x:.1f}' y='{dim_y-2:.1f}' class='sub' text-anchor='middle' font-size='7'>{op_type}</text>")
        svg.append(f"<text x='{dim_x:.1f}' y='{dim_y+6:.1f}' class='sub' text-anchor='middle' font-size='6'>{width_ft_in}</text>")

    def _mm_to_feet_inches_str(self, mm: float) -> str:
        """Convert millimeters to feet-inches string format."""
        total_inches = mm / 25.4
        feet = int(total_inches // 12)
        inches = total_inches - feet * 12
        inches_rounded = int(round(inches))
        if inches_rounded == 12:
            feet += 1
            inches_rounded = 0
        return f"{feet}'-{inches_rounded}\""

    def _add_code_compliance_indicators(self, svg: List[str], T) -> None:
        """Add building code compliance indicators."""
        # Find egress paths and exits
        self._add_egress_indicators(svg, T)
        
        # Add accessibility indicators
        self._add_accessibility_indicators(svg, T)
        
        # Add fire safety elements
        self._add_fire_safety_indicators(svg, T)
        
        # Add code compliance legend
        self._add_code_compliance_legend(svg)

    def _add_egress_indicators(self, svg: List[str], T) -> None:
        """Add egress path indicators and exit signs."""
        # Find exterior doors (main exits)
        exit_doors = []
        for op in self.openings:
            if op.type == "door":
                wll = next((w for w in self.walls if w.id == op.wall_id), None)
                if wll and wll.type == "exterior":
                    # Calculate door center
                    (ox1, oy1), (ox2, oy2) = self._opening_span(wll, op)
                    door_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
                    exit_doors.append((door_center, op))
        
        # Draw egress paths from bedrooms to exits
        for sp in self.spaces:
            name_lower = (sp.name or "").lower()
            if any(k in name_lower for k in ["bed", "master"]):
                # Calculate room center
                if sp.boundary and len(sp.boundary) >= 3:
                    cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
                    cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
                    
                    # Find nearest exit
                    if exit_doors:
                        nearest_exit = min(exit_doors, key=lambda x: ((x[0][0] - cx)**2 + (x[0][1] - cy)**2)**0.5)
                        exit_x, exit_y = nearest_exit[0]
                        
                        # Draw egress path
                        X1, Y1 = T(cx, cy)
                        X2, Y2 = T(exit_x, exit_y)
                        
                        # Dashed line for egress path
                        svg.append(f"<line x1='{X1:.1f}' y1='{Y1:.1f}' x2='{X2:.1f}' y2='{Y2:.1f}' stroke='#FF4444' stroke-width='2' stroke-dasharray='8,4' opacity='0.8'/>")
                        
                        # Egress arrow at exit
                        angle = math.atan2(Y2 - Y1, X2 - X1)
                        arrow_size = 8
                        ax = X2 - arrow_size * math.cos(angle)
                        ay = Y2 - arrow_size * math.sin(angle)
                        px = arrow_size * 0.5 * math.cos(angle + math.pi/2)
                        py = arrow_size * 0.5 * math.sin(angle + math.pi/2)
                        
                        svg.append(f"<path d='M {X2:.1f} {Y2:.1f} L {ax+px:.1f} {ay+py:.1f} L {ax-px:.1f} {ay-py:.1f} Z' fill='#FF4444'/>")
        
        # Mark exit doors with EXIT signs
        for (door_center, op) in exit_doors:
            X, Y = T(*door_center)
            svg.append(f"<rect x='{X-20:.1f}' y='{Y-30:.1f}' width='40' height='12' fill='#FF0000' stroke='white' stroke-width='1'/>")
            svg.append(f"<text x='{X:.1f}' y='{Y-22:.1f}' class='label' text-anchor='middle' fill='white' font-size='8' font-weight='bold'>EXIT</text>")

    def _add_accessibility_indicators(self, svg: List[str], T) -> None:
        """Add accessibility compliance indicators."""
        # Check door widths for ADA compliance (min 32" clear width)
        for op in self.openings:
            if op.type == "door":
                wll = next((w for w in self.walls if w.id == op.wall_id), None)
                if wll:
                    (ox1, oy1), (ox2, oy2) = self._opening_span(wll, op)
                    door_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
                    X, Y = T(*door_center)
                    
                    # Check if door is ADA compliant (32" = 813mm minimum)
                    if op.width >= 813:  # ADA compliant
                        svg.append(f"<circle cx='{X+15:.1f}' cy='{Y-15:.1f}' r='8' fill='#0066CC' stroke='white' stroke-width='1'/>")
                        svg.append(f"<text x='{X+15:.1f}' y='{Y-11:.1f}' class='label' text-anchor='middle' fill='white' font-size='6' font-weight='bold'>♿</text>")
        
        # Mark accessible bathrooms
        for sp in self.spaces:
            name_lower = (sp.name or "").lower()
            if any(k in name_lower for k in ["bath", "toilet", "wc"]) and sp.boundary:
                # Calculate room area for accessibility requirements
                def poly_area(pts):
                    a = 0.0
                    for i in range(len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[(i + 1) % len(pts)]
                        a += x1 * y2 - x2 * y1
                    return abs(a) * 0.5
                
                area_mm2 = poly_area(sp.boundary)
                # ADA requires min 30" x 48" (762mm x 1219mm) clear floor space = 0.93 sq m
                if area_mm2 >= 930000:  # Accessible bathroom size
                    cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
                    cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
                    X, Y = T(cx, cy)
                    svg.append(f"<circle cx='{X+20:.1f}' cy='{Y-20:.1f}' r='6' fill='#0066CC'/>")
                    svg.append(f"<text x='{X+20:.1f}' y='{Y-16:.1f}' class='label' text-anchor='middle' fill='white' font-size='10'>♿</text>")

    def _add_fire_safety_indicators(self, svg: List[str], T) -> None:
        """Add fire safety and smoke detection indicators."""
        # Add smoke detector symbols in key locations
        smoke_detector_locations = []
        
        # Smoke detectors in bedrooms and hallways
        for sp in self.spaces:
            name_lower = (sp.name or "").lower()
            if any(k in name_lower for k in ["bed", "master", "hall", "corridor"]) and sp.boundary:
                cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
                cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
                smoke_detector_locations.append((cx, cy))
        
        # Also add in living areas
        for sp in self.spaces:
            name_lower = (sp.name or "").lower()
            if any(k in name_lower for k in ["living", "family", "great"]) and sp.boundary:
                cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
                cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
                smoke_detector_locations.append((cx, cy))
        
        # Draw smoke detector symbols
        for (sx, sy) in smoke_detector_locations:
            X, Y = T(sx, sy)
            svg.append(f"<circle cx='{X:.1f}' cy='{Y:.1f}' r='6' fill='#FFD700' stroke='#B8860B' stroke-width='1'/>")
            svg.append(f"<text x='{X:.1f}' y='{Y+2:.1f}' class='label' text-anchor='middle' font-size='8'>S</text>")
        
        # Mark fire extinguisher locations (typically in kitchen and garage)
        for sp in self.spaces:
            name_lower = (sp.name or "").lower()
            if any(k in name_lower for k in ["kitchen", "garage"]) and sp.boundary:
                cx = sum(p[0] for p in sp.boundary) / len(sp.boundary)
                cy = sum(p[1] for p in sp.boundary) / len(sp.boundary)
                X, Y = T(cx, cy)
                # Position near wall
                X += 25  # Offset from center
                svg.append(f"<rect x='{X-5:.1f}' y='{Y-8:.1f}' width='10' height='16' fill='#FF0000' stroke='white' stroke-width='1'/>")
                svg.append(f"<text x='{X:.1f}' y='{Y+2:.1f}' class='label' text-anchor='middle' fill='white' font-size='8' font-weight='bold'>FE</text>")

    def _add_code_compliance_legend(self, svg: List[str]) -> None:
        """Add legend for code compliance symbols."""
        legend_x, legend_y = 50, 50
        
        svg.append(f"<g id='compliance-legend'>")
        svg.append(f"<rect x='{legend_x-10}' y='{legend_y-10}' width='200' height='120' fill='white' stroke='#333' stroke-width='1' opacity='0.95'/>")
        svg.append(f"<text x='{legend_x}' y='{legend_y+5}' class='label' font-size='10' font-weight='bold'>CODE COMPLIANCE</text>")
        
        # Egress path
        svg.append(f"<line x1='{legend_x}' y1='{legend_y+18}' x2='{legend_x+25}' y2='{legend_y+18}' stroke='#FF4444' stroke-width='2' stroke-dasharray='8,4'/>")
        svg.append(f"<text x='{legend_x+30}' y='{legend_y+22}' class='sub' font-size='8'>Egress Path</text>")
        
        # Exit sign
        svg.append(f"<rect x='{legend_x}' y='{legend_y+28}' width='20' height='6' fill='#FF0000'/>")
        svg.append(f"<text x='{legend_x+10}' y='{legend_y+32}' class='label' text-anchor='middle' fill='white' font-size='4'>EXIT</text>")
        svg.append(f"<text x='{legend_x+25}' y='{legend_y+32}' class='sub' font-size='8'>Exit Door</text>")
        
        # ADA symbol
        svg.append(f"<circle cx='{legend_x+8}' cy='{legend_y+45}' r='6' fill='#0066CC'/>")
        svg.append(f"<text x='{legend_x+8}' y='{legend_y+49}' class='label' text-anchor='middle' fill='white' font-size='8'>♿</text>")
        svg.append(f"<text x='{legend_x+20}' y='{legend_y+49}' class='sub' font-size='8'>ADA Compliant</text>")
        
        # Smoke detector
        svg.append(f"<circle cx='{legend_x+8}' cy='{legend_y+60}' r='6' fill='#FFD700' stroke='#B8860B' stroke-width='1'/>")
        svg.append(f"<text x='{legend_x+8}' y='{legend_y+63}' class='label' text-anchor='middle' font-size='8'>S</text>")
        svg.append(f"<text x='{legend_x+20}' y='{legend_y+63}' class='sub' font-size='8'>Smoke Detector</text>")
        
        # Fire extinguisher
        svg.append(f"<rect x='{legend_x+5}' y='{legend_y+72}' width='6' height='10' fill='#FF0000'/>")
        svg.append(f"<text x='{legend_x+8}' y='{legend_y+79}' class='label' text-anchor='middle' fill='white' font-size='6'>FE</text>")
        svg.append(f"<text x='{legend_x+20}' y='{legend_y+79}' class='sub' font-size='8'>Fire Extinguisher</text>")
        
        # Compliance note (escape ampersand)
        svg.append(f"<text x='{legend_x}' y='{legend_y+100}' class='sub' font-size='7' font-style='italic'>Per NBC 2016 &amp; Accessibility</text>")
        svg.append(f"<text x='{legend_x}' y='{legend_y+110}' class='sub' font-size='7' font-style='italic'>Standards for Design</text>")
        
        svg.append(f"</g>")


def export_plan_svg(input_path: str, output_path: str, width: int = 1800, height: int = 1200, sheet_mode: str = "floor") -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    renderer = CADRenderer(plan, sheet_mode=sheet_mode)
    svg = renderer.render(width, height)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(svg, encoding="utf-8")
    print(f"✅ Plan SVG written to {output_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render HouseBrain Plan JSON to SVG")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--mode", default="floor", choices=["floor", "rcp", "power", "plumbing"])
    args = ap.parse_args()
    export_plan_svg(args.input, args.output, args.width, args.height, args.mode)


