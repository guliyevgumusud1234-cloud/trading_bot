import json
import os

# Açık pozisyonları kaydetmek için dosya
def save_open_position(position):
    if os.path.exists('logs/open_positions.json'):
        with open('logs/open_positions.json', 'r') as file:
            positions = json.load(file)
    else:
        positions = []
    
    positions.append(position)
    
    with open('logs/open_positions.json', 'w') as file:
        json.dump(positions, file)

# Pozisyon kapama
def close_position(position_id):
    with open('logs/open_positions.json', 'r') as file:
        positions = json.load(file)
    
    positions = [pos for pos in positions if pos['id'] != position_id]
    
    with open('logs/open_positions.json', 'w') as file:
        json.dump(positions, file)