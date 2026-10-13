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


class DisplayPlayer:
    """Metric XY/XZ panels with bounded, schematic entity glyphs."""
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.game_over_font = pygame.font.Font(None, 64)
        self.label_font = pygame.font.Font(None, 17)
        self.badge_font = pygame.font.Font(None, 15)
        self.transparent_layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        resources = Path(__file__).resolve().parents[1] / 'resources'
        self.target_image = pygame.image.load(str(resources / 'target.png'))
        self.dead_target_image = pygame.image.load(str(resources / 'target_dead.png'))
        self.result_surfaces = {1: self.game_over_font.render('Red Wins', True, RedColor),
                               -1: self.game_over_font.render('Blue Wins', True, BlueColor)}
        from had_env.config import AeroPoint
        self.configure_bounds(AeroPoint)

    def configure_bounds(self, bounds):
        self.bounds = np.asarray(bounds, float)
        raw_span = self.bounds[:, 1] - self.bounds[:, 0]
        # A constant axis has no measurable extent; reserve a readable plane
        # panel instead of allowing its clip rectangle to round to zero pixels.
        span = np.where(raw_span > 0, raw_span, max(float(raw_span.max()), 1.) * .1)
        width, height = self.screen.get_size()
        scale = min((width-100)/span[0], (height-148)/(span[1]+span[2]))
        panel_width = scale*span[0]
        left = (width-panel_width)/2
        xy_height, xz_height = scale*span[1], scale*span[2]
        self._panels = {'xy': (left, 72., panel_width, xy_height),
                        'xz': (left, 108.+xy_height, panel_width, xz_height)}
        self.panel_rects = {key: pygame.Rect(round(x), round(y), round(w), round(h))
                            for key, (x,y,w,h) in self._panels.items()}
        self.interaction_length_range = [left, left+panel_width]
        self.interaction_width_range = [72., 72.+xy_height]
        self.interaction_height_range = [108.+xy_height, 108.+xy_height+xz_height]

    def project_position(self, position, projection='xy'):
        x, y, width, height = self._panels[projection]
        return x+float(position[0])*width, y+(1-float(position[1 if projection=='xy' else 2]))*height

    def update(self, agent_info_list, game_result, metadata=None):
        if not self.running:
            return
        self.handle_events()
        if self.running:
            self.draw(agent_info_list, game_result, metadata)
            pygame.display.flip()

    def draw(self, agent_info_list, game_result, metadata=None):
        if metadata is not None:
            self.configure_bounds(metadata['world_bounds'])
        layer = self.transparent_layer
        layer.fill((0,0,0,0))
        self.draw_borders()
        targets, red, blue = agent_info_list
        self.draw_targets(layer, targets)
        self.draw_agents(layer, red, 'Red')
        self.draw_agents(layer, blue, 'Blue')
        self.screen.blit(layer, (0,0))
        self.draw_labels(metadata)
        self.display_result(game_result)

    def _entity_label(self, surface, label, location, color, projection):
        image = self.badge_font.render(label, True, (255,255,255))
        rect = image.get_rect(topleft=(location[0]+18, location[1]-18))
        rect.clamp_ip(self.panel_rects[projection].inflate(-6,-6))
        pygame.draw.rect(surface, color, rect.inflate(4,2), border_radius=2)
        surface.blit(image, rect)

    def draw_targets(self, surface, targets_info_n, size=12):
        previous_clip = surface.get_clip()
        try:
            for projection in ('xy','xz'):
                surface.set_clip(self.panel_rects[projection].clip(previous_clip))
                for row in targets_info_n:
                    center = self.project_position(row['position'], projection)
                    alive = row.get('alive')
                    color = RedColor if alive is True else DeadAgentColor
                    if projection == 'xy':
                        image = self.target_image if alive is True else self.dead_target_image
                        surface.blit(image, image.get_rect(center=(round(center[0]),round(center[1]))))
                    else:
                        pygame.draw.circle(surface, color, center, int(size/1.5), 0 if alive is True else 2)
                        if alive is False:
                            x,y=center
                            pygame.draw.line(surface,color,(x-5,y-5),(x+5,y+5),2)
                    if 'id' in row:
                        self._entity_label(surface, 'T'+str(row['id'])+(' ?' if alive is None else ''), center, color, projection)
        finally:
            surface.set_clip(previous_clip)

    def draw_agents(self, surface, agents_info_n, color, size=12):
        team_color = RedColor if color == 'Red' else BlueColor
        previous_clip = surface.get_clip()
        try:
            for projection in ('xy','xz'):
                surface.set_clip(self.panel_rects[projection].clip(previous_clip))
                for row in agents_info_n:
                    location = self.project_position(row['position'], projection)
                    draw_color = team_color if row.get('alive') else DeadAgentColor
                    model = row.get('env_agent_type','particle')
                    glyph_size = max(12,size) if projection=='xy' else max(10,size/2)
                    if model in ('UAV_fixedwing','UAV_quadrotor'):
                        geometry = aircraft_geometry(row, projection, glyph_size, location)
                        for polygon in geometry['polygons']:
                            pygame.draw.polygon(surface,draw_color,polygon)
                        for start,end in geometry['lines']:
                            pygame.draw.line(surface,draw_color,start,end,2)
                        for point,radius in geometry['circles']:
                            pygame.draw.circle(surface,draw_color,point,max(2,round(radius)),1)
                        if geometry['nose'] is not None:
                            pygame.draw.line(surface,draw_color,location,geometry['nose'],2)
                            pygame.draw.circle(surface,(255,255,255),geometry['nose'],2)
                        elif geometry['toward']:
                            pygame.draw.circle(surface,draw_color,location,2)
                        else:
                            x,y=location
                            pygame.draw.line(surface,draw_color,(x-3,y-3),(x+3,y+3),2)
                            pygame.draw.line(surface,draw_color,(x-3,y+3),(x+3,y-3),2)
                        x,y=location
                        for key,label,offset in (('bank','R',-18),('pitch','P',6)):
                            surface.blit(self.badge_font.render(label,True,draw_color),(x+offset,y+16))
                            points=geometry['indicators'].get(key)
                            if points:
                                pygame.draw.line(surface,draw_color,*points,1)
                            else:
                                surface.blit(self.badge_font.render('?',True,draw_color),(x+offset+9,y+16))
                        label = (color[0]+str(row['id'])+' ' if 'id' in row else '')+role_badge(row)
                        self._entity_label(surface,label,location,draw_color,projection)
                    else:
                        particle_size = size if projection=='xy' else size/2
                        if row.get('type')=='Attack':
                            pygame.draw.circle(surface,draw_color,location,int(particle_size))
                        else:
                            velocity=row.get('velocity') or (0,0,0)
                            axis=1 if projection=='xy' else 2
                            angle=-math.atan2(velocity[axis],velocity[0])
                            draw_isosceles_triangle(surface,particle_size,location,angle,draw_color,
                                45 if row.get('type')=='Disturb' else 36)
                        if 'id' in row:
                            self._entity_label(surface,color[0]+str(row['id']),location,draw_color,projection)
        finally:
            surface.set_clip(previous_clip)

    def draw_borders(self):
        for key,rect in self.panel_rects.items():
            pygame.draw.rect(self.screen,BorderColor,rect,2)
            for fraction in (.25,.5,.75):
                x,y,w,h=self._panels[key]
                pygame.draw.line(self.screen,(213,223,232),(x+fraction*w,y),(x+fraction*w,y+h),1)
                pygame.draw.line(self.screen,(213,223,232),(x,y+fraction*h),(x+w,y+fraction*h),1)

    def draw_labels(self, metadata=None):
        ink=(36,52,71)
        def text(message,location):
            available=self.screen.get_width()-int(location[0])-20
            while self.label_font.size(message)[0]>available and len(message)>4:
                message=message[:-5]+'...'
            self.screen.blit(self.label_font.render(message,True,ink),location)
        for projection,axis in (('xy',1),('xz',2)):
            rect=self.panel_rects[projection]
            b=self.bounds
            text(f'{projection.upper()} | x {b[0,0]:g}..{b[0,1]:g} / {"y" if axis==1 else "z"} {b[axis,0]:g}..{b[axis,1]:g} m | equal scale',
                 (40,rect.top-20))
        text(ROLE_LEGEND,(40,self.screen.get_height()-18))
        if metadata is None:
            return
        counts=metadata['live_counts']
        text(f"{metadata['task_mode'].upper()} | {metadata.get('env_agent_type','')} | step {metadata['step']} | live R {counts['red']} / B {counts['blue']} | damage {metadata['step_target_damage']:.3f} / {metadata['target_damage']:.3f}",(40,8))
        def hp(value):
            return '<0.001' if 0<value<.001 else f'{value:.2f}'
        text('  '.join(f"T{row['id']} HP {hp(row['health'])}/{row['max_health']:.2f}" for row in metadata['targets']),(40,28))
        text('Live: HP > 0 | Survival terminal threshold: HP < 0.001 | z up | schematic aircraft',
             (40,self.screen.get_height()-36))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type==QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
                self.running=False
        if not self.running:
            self.close_window()

    def display_result(self, game_result):
        message=self.result_surfaces.get(game_result)
        if message is not None:
            self.screen.blit(message,message.get_rect(center=(self.screen.get_width()/2,self.screen.get_height()/2)))

    def wait_for_result(self, game_result):
        return None

    @staticmethod
    def close_window():
        pygame.display.quit()
