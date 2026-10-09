import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, QUIT
from had_env.config import (
    BlueColor,
    BorderColor,
    DeadAgentColor,
    RedColor,
    ScreenHeight,
    ScreenLength,
    ScreenWidth,
)
from pathlib import Path

import math
from .glyphs import aircraft_geometry, role_badge, ROLE_LEGEND
from had_env.geometry import get_area_point


def draw_isosceles_triangle(screen, size, center, angle=0., color=(255, 0, 0), top_angle=45):
    """
    绘制一个等腰三角形，顶角朝向指定的角度（angle=0 时表示向右）。

    参数：
    - screen: Pygame 屏幕对象。
    - size: 三角形的高度（从顶点到底边中心）。
    - center: 中心坐标 (x, y)，表示三角形质心。
    - angle: 顶角朝向角度，单位为弧度。0 表示朝右。
    - color: 填充颜色。
    """
    size = size * 2

    half_base = size * math.tan(math.radians(top_angle / 2))

    vertex_top = (size / 2, 0)
    vertex_left = (-size / 2, -half_base)
    vertex_right = (-size / 2, half_base)

    vertices = [vertex_top, vertex_left, vertex_right]

    rotated_vertices = []
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for x, y in vertices:
        x_rot = x * cos_a - y * sin_a + center[0]
        y_rot = x * sin_a + y * cos_a + center[1]
        rotated_vertices.append((x_rot, y_rot))

    pygame.draw.polygon(screen, color, rotated_vertices)


# Base class for the display system
class DisplayPlayer:
    def __init__(self, screen):
        self.screen = screen
        self.interaction_length_range = [int(0.05 * ScreenLength), int(0.95 * ScreenLength)]
        self.interaction_width_range = [int(0.05 * ScreenWidth), int(0.95 * ScreenWidth)]
        self.interaction_height_range = [int(0.05 * ScreenHeight), int(0.95 * ScreenHeight)]
        self.running = True
        self.game_over_font = pygame.font.Font(None, 72)
        self.label_font = pygame.font.Font(None, 17)
        self.badge_font = pygame.font.Font(None, 15)
        self.transparent_layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        resource_dir = Path(__file__).resolve().parents[1] / 'resources'
        self.target_image = pygame.image.load(str(resource_dir / 'target.png'))
        self.dead_target_image = pygame.image.load(str(resource_dir / 'target_dead.png'))
        self.result_surfaces = {
            1: self.game_over_font.render("Red Wins", True, (255, 0, 0)),
            -1: self.game_over_font.render("Blue Wins", True, (0, 0, 255)),
        }

    def update(self, agent_info_list, game_result, metadata=None):
        if not self.running:
            return

        self.handle_events()
        if not self.running:
            return

        self.draw(agent_info_list, game_result, metadata)
        pygame.display.flip()

    def draw(self, agent_info_list, game_result, metadata=None):
        """Draw one frame without processing events or flipping a display.

        This is the reusable rendering primitive for both the pygame window
        and off-screen scientific episode viewers.
        """

        targets_info_n, red_agents_info_n, blue_agents_info_n = agent_info_list

        transparent_layer = self.transparent_layer
        transparent_layer.fill((0, 0, 0, 0))

        self.draw_targets(transparent_layer, targets_info_n)

        self.draw_agents(transparent_layer, red_agents_info_n, color="Red")
        self.draw_agents(transparent_layer, blue_agents_info_n, color="Blue")

        self.screen.blit(transparent_layer, (0, 0))

        self.draw_borders()
        self.draw_labels(metadata)

        self.display_result(game_result)

    def project_position(self, position, projection="xy"):
        """Normalized world position to the native XY/XZ panels (y/z up)."""
        x, y, z = position
        horizontal = get_area_point(self.interaction_length_range, x)
        if projection == "xy":
            vertical = get_area_point(self.interaction_width_range, 1-y) + self.interaction_height_range[0]
        else:
            vertical = self.interaction_width_range[1] + get_area_point(self.interaction_height_range, 1-z)
        return horizontal, vertical

    def draw_labels(self, metadata=None):
        ink = (36, 52, 71)
        def text(message, location):
            self.screen.blit(self.label_font.render(message, True, ink), location)
        text(ROLE_LEGEND, (40, self.screen.get_height()-18))
        if metadata is None:
            text("XY / m", (44, 52))
            text("XZ / m  (z up)", (44, self.screen.get_height()-38))
            return
        bounds = metadata["world_bounds"]
        counts = metadata["live_counts"]
        text(f"{metadata['task_mode'].upper()}  |  step {metadata['step']}  |  live Red {counts['red']} / Blue {counts['blue']}"
             f"  |  damage {metadata['step_target_damage']:.3f} step / {metadata['target_damage']:.3f} total", (40, 8))
        assets = "  ".join(f"T{row['id']} HP {row['health']:.2f}/{row['max_health']:.2f}" for row in metadata["targets"])
        # Keep dense target sets inside the fixed-size frame.
        limit = self.screen.get_width()-80
        while self.label_font.size(assets)[0] > limit and len(assets) > 4:
            assets = assets[:-5] + "..."
        text(assets, (40, 28))
        text(f"XY  |  x {bounds[0][0]:g}..{bounds[0][1]:g} / y {bounds[1][0]:g}..{bounds[1][1]:g} m", (44, 52))
        text(f"XZ  |  x {bounds[0][0]:g}..{bounds[0][1]:g} / z {bounds[2][0]:g}..{bounds[2][1]:g} m  (z up)",
             (44, self.screen.get_height()-38))

    def handle_events(self):
        """Handle user input events like quitting."""
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
            elif event.type == QUIT:
                self.running = False

        if not self.running:
            self.close_window()

    def draw_borders(self):
        """Draw the borders of the screen."""
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         (self.interaction_length_range[1], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         (self.interaction_length_range[0], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         (self.interaction_length_range[1], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[1], self.interaction_width_range[1] + self.interaction_height_range[1]),
                         (self.interaction_length_range[1], self.interaction_width_range[0] + self.interaction_height_range[0]),
                         2)
        pygame.draw.line(self.screen, BorderColor,
                         (self.interaction_length_range[0], self.interaction_width_range[1] + self.interaction_height_range[0]),
                         (self.interaction_length_range[1], self.interaction_width_range[1] + self.interaction_height_range[0]),
                         2)

    def draw_targets(self, surface, targets_info_n, size=12):
        for target_info in targets_info_n:
            agent_px, agent_py, agent_pz = target_info["position"]
            draw_target = self.target_image if target_info["alive"] else self.dead_target_image

            center = self.project_position(target_info["position"])
            # resized_image = pygame.transform.scale(image, (new_width, new_height))
            surface.blit(draw_target, center)
            center_z = self.project_position(target_info["position"], "xz")
            pygame.draw.circle(surface, RedColor, center_z, int(size/1.5))

    def draw_agents(self, surface, agents_info_n, color, size=12):
        draw_color = RedColor if color == "Red" else BlueColor
        for agent_info in agents_info_n:
            agent_px, agent_py, agent_pz = agent_info["position"]
            agent_vx, agent_vy, agent_vz = agent_info["velocity"]
            draw_agent_color = draw_color if agent_info["alive"] else DeadAgentColor

            center = self.project_position(agent_info["position"])
            angle = - math.atan2(agent_vy, agent_vx)

            center_z = self.project_position(agent_info["position"], "xz")
            angle_z = np.pi / 2 if agent_vz < 0 else - np.pi / 2

            model = agent_info.get("env_agent_type", "particle")
            if model in ("UAV_fixedwing", "UAV_quadrotor"):
                for projection, location, glyph_size in (("xy", center, max(12, size)), ("xz", center_z, max(10, size / 2))):
                    geometry = aircraft_geometry(agent_info, projection, glyph_size, location)
                    for polygon in geometry["polygons"]:
                        pygame.draw.polygon(surface, draw_agent_color, polygon)
                    for start, end in geometry["lines"]:
                        pygame.draw.line(surface, draw_agent_color, start, end, 2)
                    for point, radius in geometry["circles"]:
                        pygame.draw.circle(surface, draw_agent_color, point, max(2, round(radius)), 1)
                    if geometry["nose"] is not None:
                        pygame.draw.line(surface, draw_agent_color, location, geometry["nose"], 2)
                        pygame.draw.circle(surface, (255, 255, 255), geometry["nose"], 2)
                    elif geometry["toward"]:
                        pygame.draw.circle(surface, draw_agent_color, location, 2)
                    else:
                        x, y = location
                        pygame.draw.line(surface, draw_agent_color, (x-3, y-3), (x+3, y+3), 2)
                        pygame.draw.line(surface, draw_agent_color, (x-3, y+3), (x+3, y-3), 2)
                    x, y = location
                    pygame.draw.rect(surface, draw_agent_color, (x+16, y-18, 12, 12), border_radius=2)
                    surface.blit(self.badge_font.render(role_badge(agent_info), True, (255, 255, 255)), (x+19, y-18))
                    for key, label, offset in (("bank", "R", -18), ("pitch", "P", 6)):
                        surface.blit(self.badge_font.render(label, True, draw_agent_color), (x+offset, y+16))
                        points = geometry["indicators"].get(key)
                        if points:
                            pygame.draw.line(surface, draw_agent_color, *points, 1)
                        else:
                            surface.blit(self.badge_font.render("?", True, draw_agent_color), (x+offset+9, y+16))
            elif agent_info["type"] == "Attack":
                pygame.draw.circle(surface, draw_agent_color, center, int(size))
                pygame.draw.circle(surface, draw_agent_color, center_z, int(size / 2))
            elif agent_info["type"] == "Disturb":
                draw_isosceles_triangle(surface, size=size, center=center, angle=angle, color=draw_agent_color,
                                        top_angle=45)
                draw_isosceles_triangle(surface, size=int(size / 2), center=center_z, angle=angle_z,
                                        color=draw_agent_color, top_angle=45)
            else:
                draw_isosceles_triangle(surface, size=size, center=center, angle=angle, color=draw_agent_color,
                                        top_angle=36)
                draw_isosceles_triangle(surface, size=int(size / 2), center=center_z, angle=angle_z,
                                        color=draw_agent_color, top_angle=36)

    def display_result(self, game_result):
        text_surface = self.result_surfaces.get(game_result)
        if text_surface is None:
            return
        text_rect = text_surface.get_rect(center=(self.screen.get_width() / 2, self.screen.get_height() / 2))
        self.screen.blit(text_surface, text_rect)

    def wait_for_result(self, game_result):
        """Compatibility no-op: the application owns timing and event polling."""
        return None

    @staticmethod
    def close_window():
        """Close the pygame window."""
        pygame.display.quit()
