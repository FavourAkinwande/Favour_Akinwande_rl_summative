from __future__ import annotations

import math
import random
import numpy as np
import pygame


class FoodDashboard:
    """Pygame dashboard wrapper for visualizing the FoodRedistributionEnv."""

    def __init__(self, width: int = 900, height: int = 650) -> None:
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Food Redistribution RL – Dashboard")
        self.clock = pygame.time.Clock()
        self.running = True

        # Layout rectangles (will be recalculated on resize)
        self._update_layout_rectangles()

        # Fonts (increased sizes for better readability)
        self.font_large = pygame.font.SysFont("poppins", 28)
        self.font_medium = pygame.font.SysFont("poppins", 20)
        self.font_small = pygame.font.SysFont("poppins", 16)

        # Bright, playful palette
        self.colors = {
            "bg_primary": (247, 249, 245),   # fresh neutral
            "bg_secondary": (233, 242, 236), # mint
            "bg_tertiary": (255, 255, 255),  # cards
            "border": (197, 214, 205),       # soft teal border
            "text_primary": (24, 52, 58),    # deep teal
            "text_secondary": (93, 119, 123),
            "accent": (22, 185, 140),        # mint green
            "warning": (255, 163, 72),       # mango
            "success": (133, 217, 78),       # lime
            "highlight": (255, 210, 92),     # sunshine
            "coral": (255, 120, 120),        # coral pop
        }

        # Particle field for animated background
        self.particles = self._spawn_particles(count=70)

        # Delivery animation state
        self.truck_anim = {
            "path_key": None,
            "active": False,
            "start_time": 0,
            "duration": 1000,  # milliseconds
            "start_pos": (0, 0),
            "end_pos": (0, 0),
            "pulse_time": 0,
        }
        self._active_env = None
        self._dragging = False

    def cleanup(self) -> None:
        """Close the pygame window."""
        pygame.quit()

    def _update_layout_rectangles(self) -> None:
        """Recalculate layout rectangles based on current window size."""
        self.header_rect = pygame.Rect(0, 0, self.width, 110)
        # Sidebar: fixed width, positioned on the right
        sidebar_width = 320
        sidebar_margin = 20
        # Center panel: takes remaining space after sidebar and margins
        center_left_margin = 24
        center_right_margin = 16
        center_width = max(320, self.width - sidebar_width - sidebar_margin - center_left_margin - center_right_margin)
        center_height = max(260, self.height - 150)  # Leave room for header and footer
        self.center_rect = pygame.Rect(center_left_margin, 130, center_width, center_height)
        # Sidebar positioning
        sidebar_height = max(260, self.height - 150)
        self.sidebar_rect = pygame.Rect(self.width - sidebar_width - sidebar_margin, 120, sidebar_width, sidebar_height)
        self.footer_rect = pygame.Rect(0, self.height - 40, self.width, 40)
        # Respawn particles in new center area
        self.particles = self._spawn_particles(count=70)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self._update_layout_rectangles()
            elif event.type == pygame.MOUSEWHEEL and self._active_env:
                if hasattr(self._active_env, "handle_camera_zoom"):
                    self._active_env.handle_camera_zoom(event.y)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging and self._active_env:
                if hasattr(self._active_env, "handle_camera_orbit"):
                    dx, dy = event.rel
                    self._active_env.handle_camera_orbit(dx, dy)

    def _blit_center_frame(self, env) -> None:
        pygame.draw.rect(self.screen, self.colors["bg_secondary"], self.center_rect, border_radius=30)
        frame = None
        if hasattr(env, "render"):
            try:
                frame = env.render(mode="rgb_array")
            except Exception as exc:
                print("env.render error:", exc)
                frame = None

        if isinstance(frame, np.ndarray):
            h, w, _ = frame.shape
            surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
            surf = pygame.transform.smoothscale(surf, (self.center_rect.width, self.center_rect.height))
            self.screen.blit(surf, self.center_rect.topleft)
        else:
            msg = self.font_small.render("Rendering unavailable", True, self.colors["text_secondary"])
            self.screen.blit(
                msg,
                (
                    self.center_rect.centerx - msg.get_width() // 2,
                    self.center_rect.centery - msg.get_height() // 2,
                ),
            )

        metrics = getattr(env, "last_info", {}) or {}
        self._draw_summary_cards(env, metrics)

    def _draw_header(self, env, episode: int, total_episodes: int) -> None:
        pygame.draw.rect(self.screen, self.colors["bg_tertiary"], self.header_rect)
        title = self.font_large.render("Food Redistribution RL – PPO Agent", True, self.colors["text_primary"])
        subtitle = self.font_small.render(
            "Simulated surplus food delivery from retailers to communities", True, self.colors["text_secondary"]
        )
        self.screen.blit(title, (30, 15))
        self.screen.blit(subtitle, (30, 60))

        # Right-hand summary
        summary_lines = [
            f"Episodes: {episode}/{total_episodes}",
            f"Current step: {getattr(env, 'time_step', 0)}/{env.config.day_length}",
            "Algorithm: PPO",
        ]
        for idx, line in enumerate(summary_lines):
            text = self.font_medium.render(line, True, self.colors["text_primary"])
            self.screen.blit(text, (self.width - 360, 25 + idx * 30))

    def _draw_sidebar(self, env, episode_reward: float) -> None:
        pygame.draw.rect(self.screen, self.colors["bg_tertiary"], self.sidebar_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.colors["border"], self.sidebar_rect, width=2, border_radius=12)

        y = self.sidebar_rect.y + 20
        section_spacing = 34

        def draw_section(title: str) -> None:
            nonlocal y
            text = self.font_medium.render(title, True, self.colors["accent"])
            self.screen.blit(text, (self.sidebar_rect.x + 20, y))
            y += section_spacing

        last_info = getattr(env, "last_info", {}) or {}

        draw_section("Agent Status")
        last_action = getattr(env, "last_action", None)
        if last_action is not None:
            r_idx, c_idx = divmod(last_action, env.num_communities)
            action_text = f"Last action: R{r_idx+1}->C{c_idx+1}"
        else:
            action_text = "Last action: N/A"
        items = [
            action_text,
            f"Last reward: {getattr(env, 'last_reward', 0.0):.3f}",
            f"Fairness: {last_info.get('fairness', 0.0):.3f}",
            f"Delivered ratio: {last_info.get('delivered_ratio', 0.0):.3f}",
            f"Episode return: {episode_reward:.3f}",
        ]
        for item in items:
            text = self.font_small.render(item, True, self.colors["text_primary"])
            self.screen.blit(text, (self.sidebar_rect.x + 20, y))
            y += section_spacing

        y += section_spacing // 2
        draw_section("System Status")
        system_items = [
            f"Total delivered: {np.sum(getattr(env, 'delivered_totals', [])):.1f}",
            f"Waste: {last_info.get('waste', 0.0):.1f}",
            f"Unmet demand: {last_info.get('unmet_demand', 0.0):.1f}",
            f"Remaining supply: {np.sum(getattr(env, 'supplies', [])):.1f}",
            f"Remaining demand: {np.sum(getattr(env, 'demands', [])):.1f}",
        ]
        for item in system_items:
            text = self.font_small.render(item, True, self.colors["text_primary"])
            self.screen.blit(text, (self.sidebar_rect.x + 20, y))
            y += section_spacing

    def _draw_footer(self) -> None:
        pygame.draw.rect(self.screen, self.colors["bg_tertiary"], self.footer_rect)
        fps = self.clock.get_fps()
        left_text = self.font_small.render(f"FPS: {fps:.1f}", True, self.colors["text_secondary"])
        legend_text = (
            "Yellow line: chosen route | Grey retailers: low supply | Orange nodes: communities with demand"
        )
        right_text = self.font_small.render(legend_text, True, self.colors["text_secondary"])
        self.screen.blit(left_text, (20, self.footer_rect.y + 12))
        self.screen.blit(right_text, (200, self.footer_rect.y + 12))

    def render_particles(self) -> None:
        clip = self.center_rect.inflate(-40, -40)
        particle_layer = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for particle in self.particles:
            particle["x"] += particle["vx"]
            particle["y"] += particle["vy"]
            if particle["x"] < clip.left:
                particle["x"] = clip.right
            elif particle["x"] > clip.right:
                particle["x"] = clip.left
            if particle["y"] < clip.top:
                particle["y"] = clip.bottom
            elif particle["y"] > clip.bottom:
                particle["y"] = clip.top
            alpha = int(80 * particle["alpha"])
            color = (255, 200, 120, alpha)
            pygame.draw.circle(
                particle_layer,
                color,
                (int(particle["x"]), int(particle["y"])),
                int(particle["size"]),
            )
        self.screen.blit(particle_layer, (0, 0))

    def render_base_layout(self, env) -> dict:
        area = self.center_rect
        pygame.draw.rect(self.screen, self.colors["bg_secondary"], area, border_radius=30)

        num_retailers = getattr(env, "num_retailers", 3)
        num_communities = getattr(env, "num_communities", 3)
        supplies = np.array(getattr(env, "supplies", np.zeros(num_retailers)))
        demands = np.array(getattr(env, "demands", np.zeros(num_communities)))
        max_supply = max(getattr(env.config, "max_supply", 1), 1)
        max_demand = max(getattr(env.config, "max_demand", 1), 1)

        # Use proportional positioning that scales with window size
        left_margin = max(70, area.width * 0.12)
        right_margin = max(90, area.width * 0.18)
        left_x = area.x + int(left_margin)
        right_x = area.x + int(area.width - right_margin)
        
        # Vertical spacing with proper padding
        top_padding = max(40, area.height * 0.1)  # 10% padding top/bottom
        bottom_padding = max(40, area.height * 0.1)
        available_height = area.height - top_padding - bottom_padding
        spacing_left = available_height / (num_retailers + 1) if num_retailers > 0 else 0
        spacing_right = available_height / (num_communities + 1) if num_communities > 0 else 0

        retailer_nodes = []
        for idx in range(num_retailers):
            y = area.y + int(top_padding + (idx + 1) * spacing_left)
            pos = (left_x, y)
            ratio = float(supplies[idx] / max_supply)
            self._draw_retailer(pos, f"R{idx+1}", ratio, supplies[idx], max_supply)
            retailer_nodes.append({"pos": pos})

        community_nodes = []
        for idx in range(num_communities):
            y = area.y + int(top_padding + (idx + 1) * spacing_right)
            pos = (right_x, y)
            ratio = float(demands[idx] / max_demand)
            self._draw_community(pos, f"C{idx+1}", ratio, demands[idx], max_demand)
            community_nodes.append({"pos": pos})

        return {"retailers": retailer_nodes, "communities": community_nodes}

    def animate_delivery(self, start_node, end_node, metrics: dict) -> None:
        if start_node is None or end_node is None:
            self.truck_anim["active"] = False
            self.truck_anim["path_key"] = None
            return

        now = pygame.time.get_ticks()
        path_key = (start_node, end_node)

        if self.truck_anim["path_key"] != path_key:
            self.truck_anim.update(
                {
                    "path_key": path_key,
                    "active": True,
                    "start_time": now,
                    "duration": 1000,
                    "start_pos": start_node,
                    "end_pos": end_node,
                    "pulse_time": 0,
                }
            )

        anim = self.truck_anim
        if not anim["active"]:
            if anim["pulse_time"] and now - anim["pulse_time"] < 400:
                self._draw_glow_circle(anim["end_pos"], (255, 214, 10), 80, 120 - (now - anim["pulse_time"]) // 4)
            return

        t = min((now - anim["start_time"]) / anim["duration"], 1.0)
        points = [self._quadratic_point(anim["start_pos"], anim["end_pos"], i / 20.0) for i in range(21)]
        neon_color = (0, 245, 212)
        pygame.draw.lines(self.screen, neon_color, False, points, 3)

        truck_pos = self._quadratic_point(anim["start_pos"], anim["end_pos"], t)
        self._draw_glow_circle(truck_pos, neon_color, 18, 50)
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            pygame.Rect(truck_pos[0] - 6, truck_pos[1] - 4, 12, 8),
            border_radius=4,
        )

        if t >= 1.0:
            anim["active"] = False
            anim["pulse_time"] = now

    def _draw_summary_cards(self, env, metrics: dict) -> None:
        """Draw cheerful supply/demand cards along the bottom of the center area."""
        card_width = 180
        card_height = 90
        spacing = 20
        start_x = self.center_rect.x + spacing
        y = self.center_rect.bottom - card_height - spacing

        cards = [
            {
                "title": "Fresh Supply",
                "value": f"{np.sum(getattr(env, 'supplies', [])):.1f}",
                "unit": "units ready",
                "color": self.colors["accent"],
                "icon": "S",
            },
            {
                "title": "Community Demand",
                "value": f"{np.sum(getattr(env, 'demands', [])):.1f}",
                "unit": "meals needed",
                "color": self.colors["warning"],
                "icon": "D",
            },
        ]

        for idx, card in enumerate(cards):
            rect = pygame.Rect(
                start_x + idx * (card_width + spacing),
                y,
                card_width,
                card_height,
            )
            pygame.draw.rect(self.screen, self.colors["bg_tertiary"], rect, border_radius=20)
            pygame.draw.rect(self.screen, self.colors["border"], rect, width=1, border_radius=20)

            circle_center = (rect.x + 30, rect.y + 30)
            pygame.draw.circle(self.screen, card["color"], circle_center, 18)
            icon_text = self.font_small.render(card["icon"], True, (255, 255, 255))
            self.screen.blit(icon_text, (circle_center[0] - icon_text.get_width() // 2, circle_center[1] - icon_text.get_height() // 2))

            title_text = self.font_small.render(card["title"], True, self.colors["text_secondary"])
            self.screen.blit(title_text, (rect.x + 60, rect.y + 14))

            value_text = self.font_large.render(card["value"], True, self.colors["text_primary"])
            self.screen.blit(value_text, (rect.x + 60, rect.y + 36))

            unit_text = self.font_small.render(card["unit"], True, self.colors["text_secondary"])
            self.screen.blit(unit_text, (rect.x + 60, rect.y + 60))

    def _spawn_particles(self, count: int) -> list:
        clip = self.center_rect
        particles = []
        for _ in range(count):
            particles.append(
                {
                    "x": random.uniform(clip.left, clip.right),
                    "y": random.uniform(clip.top, clip.bottom),
                    "vx": random.uniform(-0.2, 0.2),
                    "vy": random.uniform(0.1, 0.4),
                    "size": random.uniform(1.5, 3.5),
                    "alpha": random.uniform(0.3, 0.8),
                }
            )
        return particles

    def _draw_retailer(self, pos, label, ratio, value, max_value) -> None:
        size = 44
        base_color = self._lerp_color((146, 214, 167), (32, 181, 128), ratio)
        rect = pygame.Rect(0, 0, size, size)
        rect.center = pos
        pygame.draw.rect(self.screen, base_color, rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255, 40), rect, width=2, border_radius=10)
        text = self.font_small.render(label, True, self.colors["text_primary"])
        self.screen.blit(text, (rect.centerx - text.get_width() // 2, rect.centery - text.get_height() // 2))
        ratio_text = self.font_small.render(f"S:{int(value)}", True, self.colors["text_secondary"])
        self.screen.blit(ratio_text, (rect.centerx - ratio_text.get_width() // 2, rect.top - 28))
        bar_width = 80
        bar_rect = pygame.Rect(rect.centerx - bar_width // 2, rect.bottom + 10, bar_width, 6)
        pygame.draw.rect(self.screen, self.colors["border"], bar_rect, border_radius=3)
        fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * ratio), bar_rect.height)
        pygame.draw.rect(self.screen, self.colors["accent"], fill_rect, border_radius=3)

    def _draw_community(self, pos, label, ratio, value, max_value) -> None:
        radius = 25
        base_color = self._lerp_color((255, 210, 120), (255, 150, 90), ratio)
        pygame.draw.circle(self.screen, base_color, pos, radius)
        pygame.draw.circle(self.screen, (255, 255, 255), pos, radius, width=2)
        text = self.font_small.render(label, True, self.colors["text_primary"])
        self.screen.blit(text, (pos[0] - text.get_width() // 2, pos[1] - text.get_height() // 2))
        ratio_tag = pygame.Surface((60, 20), pygame.SRCALPHA)
        pygame.draw.rect(ratio_tag, (*self.colors["bg_tertiary"], 245), ratio_tag.get_rect(), border_radius=10)
        tag_text = self.font_small.render(f"D:{int(value)}", True, self.colors["text_secondary"])
        ratio_tag.blit(tag_text, (30 - tag_text.get_width() // 2, 10 - tag_text.get_height() // 2))
        self.screen.blit(ratio_tag, (pos[0] - 30, pos[1] - radius - 28))
        bar_width = 80
        bar_rect = pygame.Rect(pos[0] - bar_width // 2, pos[1] + radius + 8, bar_width, 6)
        pygame.draw.rect(self.screen, self.colors["border"], bar_rect, border_radius=3)
        fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * ratio), bar_rect.height)
        pygame.draw.rect(self.screen, self.colors["warning"], fill_rect, border_radius=3)

    def _draw_glow_circle(self, pos, color, radius, alpha):
        glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (*color, alpha), (radius, radius), radius)
        self.screen.blit(glow_surface, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _quadratic_point(self, start, end, t):
        control = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 90)
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t**2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t**2 * end[1]
        return (x, y)

    def _lerp_color(self, c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def render_frame(self, env, episode: int, total_episodes: int, episode_reward: float) -> bool:
        """
        Render the dashboard frame. Returns False when the window is closed.
        """
        self._active_env = env
        self._handle_events()
        if not self.running:
            return False

        self.screen.fill(self.colors["bg_primary"])
        self._draw_header(env, episode, total_episodes)
        self._blit_center_frame(env)
        self._draw_sidebar(env, episode_reward)
        self._draw_footer()
        pygame.display.flip()
        self.clock.tick(60)
        return True

