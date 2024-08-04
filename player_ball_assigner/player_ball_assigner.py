import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
  def __init__(self):
    self.max_player_ball_distance = 40

  def assign_player_ball(self, player_track, ball_bbox):
    ball_position = get_center_of_bbox(ball_bbox)

    minimum_distance = 99999
    assigned_players = -1

    # Iterate through the player_track dictionary, changed player_track to player.., defined"player"
    for player_id, player in player_track.items():
      player_bbox = player['bbox']
      distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
      distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
      distance = min(distance_left, distance_right)

      if distance < minimum_distance:
        minimum_distance = distance
        assigned_players = player_id

      if distance < self.max_player_ball_distance:
        if distance < minimum_distance:
          minimum_distance = distance
          assigned_players = player_id

    return assigned_players