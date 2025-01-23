import os
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Dict

def init_assets(base_path: str = None) -> Dict[str, bool]:
    if base_path is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    directories = [
        'data',
        'cache',
        'logs',
        'models',
        'assets',
        'assets/backgrounds',
        'assets/icons',
        'config'
    ]

    result = {
        'status': False,
        'directories': {},
        'files': {},
        'backgrounds': {}
    }

    try:
        for directory in directories:
            dir_path = os.path.join(base_path, directory)
            os.makedirs(dir_path, exist_ok=True)
            result['directories'][directory] = True

        init_default_files(base_path)
        result['files']['default_files'] = True
        
        bg_path = os.path.join(base_path, 'assets/backgrounds/default_grid.png')
        bg_success = create_grid_background(bg_path)
        result['backgrounds']['default_grid'] = bg_success

        result['status'] = all([
            all(result['directories'].values()),
            all(result['files'].values()),
            all(result['backgrounds'].values())
        ])

        return result

    except Exception as e:
        print(f"Error initializing assets: {str(e)}")
        result['error'] = str(e)
        return result

def ensure_assets_exist(required_assets: Dict[str, str]) -> bool:
    try:
        for asset_name, asset_path in required_assets.items():
            if not os.path.exists(asset_path):
                print(f"Missing required asset: {asset_name} at {asset_path}")
                return False
        return True
    except Exception as e:
        print(f"Error checking assets: {str(e)}")
        return False

def create_grid_background(output_path: str, 
                         size: Tuple[int, int] = (800, 600),
                         grid_spacing: int = 20,
                         line_color: Tuple[int, int, int] = (50, 50, 50),
                         bg_color: Tuple[int, int, int] = (30, 30, 30)) -> bool:
    if not output_path:
        print("Error: No output path specified")
        return False

    try:
        image = Image.new('RGB', size, bg_color)
        draw = ImageDraw.Draw(image)

        for x in range(0, size[0], grid_spacing):
            draw.line([(x, 0), (x, size[1])], fill=line_color, width=1)

        for y in range(0, size[1], grid_spacing):
            draw.line([(0, y), (size[0], y)], fill=line_color, width=1)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        image.save(output_path)
        
        if os.path.exists(output_path):
            print(f"Successfully created grid background at {output_path}")
            return True
        else:
            print(f"Failed to verify grid background at {output_path}")
            return False

    except Exception as e:
        print(f"Error creating grid background: {str(e)}")
        return False

def init_default_files(base_path: str) -> None:
    default_files = {
        'data/blacklist_addresses.json': [],
        'data/known_exchanges.json': {
            'Ethereum': [],
            'BSC': [],
            'Tron': []
        },
        'data/known_patterns.json': {
            'mixing_services': [],
            'high_risk_addresses': [],
            'temporal_patterns': [],
            'value_patterns': []
        },
        'config/api_keys.json': {
            'etherscan': '',
            'bscscan': '',
            'trongrid': ''
        }
    }

    for file_path, default_content in default_files.items():
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                json.dump(default_content, f, indent=4)

def create_custom_background(output_path: str,
                           pattern_type: str = 'hex',
                           size: Tuple[int, int] = (800, 600),
                           color_scheme: Tuple[Tuple[int, int, int], ...] = None) -> bool:
    try:
        if color_scheme is None:
            color_scheme = ((30, 30, 30), (40, 40, 40), (50, 50, 50))

        image = Image.new('RGB', size, color_scheme[0])
        draw = ImageDraw.Draw(image)

        if pattern_type == 'hex':
            _draw_hex_pattern(draw, size, color_scheme)
        elif pattern_type == 'dots':
            _draw_dot_pattern(draw, size, color_scheme)
        elif pattern_type == 'circuit':
            _draw_circuit_pattern(draw, size, color_scheme)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        image.save(output_path)
        return True

    except Exception as e:
        print(f"Error creating custom background: {str(e)}")
        return False

def _draw_hex_pattern(draw: ImageDraw, 
                     size: Tuple[int, int],
                     colors: Tuple[Tuple[int, int, int], ...]) -> None:
    hex_size = 30
    offset = hex_size * 0.866
    
    for y in range(-hex_size, size[1] + hex_size, int(hex_size * 1.5)):
        for x in range(-hex_size, size[0] + hex_size, int(offset * 2)):
            offset_y = offset if (x // int(offset * 2)) % 2 else 0
            points = _calculate_hex_points((x, y + offset_y), hex_size)
            draw.polygon(points, fill=colors[1], outline=colors[2])

def _draw_dot_pattern(draw: ImageDraw,
                     size: Tuple[int, int],
                     colors: Tuple[Tuple[int, int, int], ...]) -> None:
    spacing = 20
    radius = 3
    
    for y in range(0, size[1], spacing):
        for x in range(0, size[0], spacing):
            offset = spacing/2 if (y // spacing) % 2 else 0
            draw.ellipse([x + offset - radius, y - radius, 
                         x + offset + radius, y + radius],
                        fill=colors[1])

def _draw_circuit_pattern(draw: ImageDraw,
                         size: Tuple[int, int],
                         colors: Tuple[Tuple[int, int, int], ...]) -> None:
    spacing = 40
    line_width = 2
    
    for x in range(0, size[0], spacing):
        for y in range(0, size[1], spacing):
            if np.random.random() > 0.5:
                draw.line([(x, y), (x + spacing, y)], 
                         fill=colors[1], width=line_width)
            if np.random.random() > 0.5:
                draw.line([(x, y), (x, y + spacing)],
                         fill=colors[1], width=line_width)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                        fill=colors[2])

def _calculate_hex_points(center: Tuple[int, int], size: int) -> list:
    x, y = center
    angles = [30, 90, 150, 210, 270, 330]
    return [(x + size * np.cos(np.radians(a)),
             y + size * np.sin(np.radians(a))) for a in angles]
