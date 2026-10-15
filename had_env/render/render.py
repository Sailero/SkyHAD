"""Pygame XY (top) and XZ (side) projections with equal metric scale."""
from collections import deque
from pathlib import Path

import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, QUIT

from .glyphs import aircraft_geometry, particle_triangle, role_badge
from .style import BlueColor, BorderColor, DeadAgentColor, GridColor, InkColor, RedColor, TargetRingColor

TRAIL_LENGTH = 20


class DisplayPlayer:
    """Draw one simulation frame onto a Pygame surface (window or offscreen)."""

    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.title_font = pygame.font.Font(None, 64)
        self.label_font = pygame.font.Font(None, 17)
        self.badge_font = pygame.font.Font(None, 15)
        self.layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        resources = Path(__file__).resolve().parents[1] / 'resources'
        self.target_image = pygame.image.load(str(resources / 'target.png'))
        self.dead_target_image = pygame.image.load(str(resources / 'target_dead.png'))
        self.trails, self._last_step = {}, None
        self.configure_bounds([[-2500., 2500.], [-2500., 2500.], [0., 2500.]])

    def configure_bounds(self, bounds):
        self.bounds = np.asarray(bounds, float)
        raw_span = self.bounds[:, 1] - self.bounds[:, 0]
        # A constant axis has no extent; reserve a readable plane panel for it.
        span = np.where(raw_span > 0, raw_span, max(float(raw_span.max()), 1.) * .1)
        width, height = self.screen.get_size()
        self.scale = min((width-100)/span[0], (height-148)/(span[1]+span[2]))  # pixels per metre
        panel_width = self.scale*span[0]
        left = (width-panel_width)/2
        xy_height, xz_height = self.scale*span[1], self.scale*span[2]
        self._panels = {'xy': (left, 72., panel_width, xy_height),
                        'xz': (left, 108.+xy_height, panel_width, xz_height)}
        self.panel_rects = {key: pygame.Rect(round(x), round(y), round(w), round(h))
                            for key, (x, y, w, h) in self._panels.items()}

    def project_position(self, position, projection='xy'):
        """Map a normalized [0, 1]^3 position to panel pixels."""
        x, y, width, height = self._panels[projection]
        return x+float(position[0])*width, y+(1-float(position[1 if projection == 'xy' else 2]))*height

    def update(self, frame, game_result, metadata=None):
        if not self.running:
            return
        self.handle_events()
        if self.running:
            self.draw(frame, game_result, metadata)
            pygame.display.flip()

    def draw(self, frame, game_result, metadata=None):
        if metadata is not None:
            self.configure_bounds(metadata['world_bounds'])
            self._record_trails(frame, metadata['step'])
        self.layer.fill((0, 0, 0, 0))
        self.draw_borders()
        targets, red, blue = frame
        self.draw_trails(self.layer)
        self.draw_targets(self.layer, targets, (metadata or {}).get('fire_range'))
        self.draw_agents(self.layer, red, 'Red')
        self.draw_agents(self.layer, blue, 'Blue')
        self.screen.blit(self.layer, (0, 0))
        self.draw_labels(metadata)
        self.display_result(game_result)

    def _record_trails(self, frame, step):
        if self._last_step is None or step < self._last_step or step == 0:
            self.trails = {}
        if step != self._last_step:
            for row in (*frame[1], *frame[2]):
                if row.get('alive'):
                    self.trails.setdefault(row['id'], deque(maxlen=TRAIL_LENGTH)).append(
                        (tuple(row['position']), row.get('team', 'Red')))
        self._last_step = step

    def _clipped(self, surface, projection):
        surface.set_clip(self.panel_rects[projection])

    def _entity_label(self, surface, label, location, color, projection):
        image = self.badge_font.render(label, True, (255, 255, 255))
        rect = image.get_rect(topleft=(location[0]+12, location[1]-17))
        rect.clamp_ip(self.panel_rects[projection].inflate(-6, -6))
        pygame.draw.rect(surface, color, rect.inflate(4, 2), border_radius=2)
        surface.blit(image, rect)

    def draw_trails(self, surface):
        for projection in ('xy', 'xz'):
            self._clipped(surface, projection)
            for points in self.trails.values():
                if len(points) > 1:
                    color = (RedColor if points[-1][1] == 'Red' else BlueColor)[:3] + (90,)
                    pygame.draw.lines(surface, color, False,
                                      [self.project_position(p, projection) for p, _ in points], 2)
        surface.set_clip(None)

    def draw_targets(self, surface, targets, fire_range=None, size=12):
        for projection in ('xy', 'xz'):
            self._clipped(surface, projection)
            for row in targets:
                center = self.project_position(row['position'], projection)
                alive = row.get('alive') is True
                if alive and fire_range:
                    pygame.draw.circle(surface, TargetRingColor, center, max(3, round(fire_range*self.scale)), 1)
                if projection == 'xy':
                    image = self.target_image if alive else self.dead_target_image
                    surface.blit(image, image.get_rect(center=(round(center[0]), round(center[1]))))
                else:
                    color = RedColor if alive else DeadAgentColor
                    pygame.draw.circle(surface, color, center, int(size/1.5), 0 if alive else 2)
                if 'label' in row:
                    self._entity_label(surface, row['label'], center, RedColor if alive else DeadAgentColor,
                                       projection)
        surface.set_clip(None)

    def draw_agents(self, surface, agents, team, size=12):
        team_color = RedColor if team == 'Red' else BlueColor
        for projection in ('xy', 'xz'):
            self._clipped(surface, projection)
            for row in agents:
                location = self.project_position(row['position'], projection)
                color = team_color if row.get('alive') else DeadAgentColor
                geometry = aircraft_geometry(row, projection, size, location)
                if geometry is not None:
                    outline = tuple(int(c*.6) for c in color[:3])
                    for polygon in geometry['polygons']:
                        pygame.draw.polygon(surface, color, polygon)
                        pygame.draw.polygon(surface, outline, polygon, 1)
                    for points, closed in zip(geometry['lines'], geometry['closed']):
                        pygame.draw.lines(surface, outline, closed, points, 2 if not closed else 1)
                    pygame.draw.circle(surface, (255, 255, 255), geometry['nose'], 2)
                elif row.get('type') == 'Attack':
                    pygame.draw.circle(surface, color, location, size if projection == 'xy' else size/2)
                else:
                    pygame.draw.polygon(surface, color, particle_triangle(
                        location, row.get('velocity') or (1, 0, 0), size/2 if projection == 'xy' else size/4,
                        projection, 45 if row.get('type') == 'Disturb' else 36))
                if 'label' in row:
                    badge = role_badge(row)
                    self._entity_label(surface, row['label'] + (' ' + badge if badge else ''), location, color,
                                       projection)
        surface.set_clip(None)

    def draw_borders(self):
        for key, rect in self.panel_rects.items():
            x, y, w, h = self._panels[key]
            for fraction in (.25, .5, .75):
                pygame.draw.line(self.screen, GridColor, (x+fraction*w, y), (x+fraction*w, y+h), 1)
                pygame.draw.line(self.screen, GridColor, (x, y+fraction*h), (x+w, y+fraction*h), 1)
            pygame.draw.rect(self.screen, BorderColor, rect, 2)

    def draw_labels(self, metadata=None):
        def text(message, location):
            available = self.screen.get_width()-int(location[0])-20
            while self.label_font.size(message)[0] > available and len(message) > 4:
                message = message[:-5]+'...'
            self.screen.blit(self.label_font.render(message, True, InkColor), location)
        b = self.bounds
        for projection, axis, name in (('xy', 1, 'top view XY'), ('xz', 2, 'side view XZ, z up')):
            text(f'{name} | x {b[0, 0]:g}..{b[0, 1]:g} m, {"y" if axis == 1 else "z"} '
                 f'{b[axis, 0]:g}..{b[axis, 1]:g} m', (40, self.panel_rects[projection].top-20))
        text('Rings: target fire range | trails: last 20 s | S scout, D disturber | grey: destroyed',
             (40, self.screen.get_height()-22))
        if metadata is None:
            return
        counts = metadata['live_counts']
        text(f"{metadata['task_mode'].upper()} | {metadata['env_agent_type']} | t = {metadata['sim_time']:g} s | "
             f"alive Red {counts['red']} / Blue {counts['blue']} | target damage {metadata['step_target_damage']:.3f} "
             f"this step, {metadata['target_damage']:.3f} total", (40, 8))
        text('  '.join(f"{row['label']} HP {row['health']:.2f}/{row['max_health']:.2f}"
                       for row in metadata['targets']), (40, 28))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.running = False
        if not self.running:
            self.close_window()

    def display_result(self, game_result):
        message = {1: ('Red wins', RedColor), -1: ('Blue wins', BlueColor)}.get(game_result)
        if message is None:
            return
        image = self.title_font.render(message[0], True, message[1])
        rect = image.get_rect(center=(self.screen.get_width()/2, self.screen.get_height()/2))
        box = pygame.Surface(rect.inflate(32, 16).size, pygame.SRCALPHA)
        box.fill((255, 255, 255, 215))
        self.screen.blit(box, rect.inflate(32, 16))
        self.screen.blit(image, rect)

    @staticmethod
    def close_window():
        pygame.display.quit()
