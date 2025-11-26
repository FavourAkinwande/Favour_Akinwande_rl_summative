"""ModernGL renderer for the FoodRedistributionEnv world scene.

Install prerequisites:
    pip install moderngl Pillow

The FoodRedistributionEnv owns an instance of OpenGLFoodScene and forwards the
latest supplies/demands/metrics each step. The scene renders an off-screen
framebuffer that the existing Pygame dashboard reuses as an RGB frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def perspective(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create a perspective projection matrix."""
    f = 1.0 / np.tan(fov / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Classic look-at view matrix."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


@dataclass
class RouteAnimation:
    active: bool = False
    start_time: float = 0.0
    duration: float = 1.1
    start: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    end: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


class OpenGLFoodScene:
    """Standalone ModernGL renderer used by the Gym environment."""

    def __init__(self, width: int = 900, height: int = 600) -> None:
        self.width = width
        self.height = height
        self.ctx = moderngl.create_standalone_context()
        self.fbo = self.ctx.simple_framebuffer((width, height), components=3)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.color_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        self.line_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_position;
                uniform mat4 mvp;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform vec3 color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        self.billboard_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                out vec2 uv;
                void main() {
                    uv = in_pos * 0.5 + 0.5;
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 uv;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(tex, uv);
                }
            """,
        )

        self.cube_vao = self._create_cube()
        self.cylinder_vao = self._create_cylinder()
        self.quad_vao = self._create_billboard()

        try:
            self.font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            self.font = ImageFont.load_default()

        self.palette = {
            "bg": (0.95, 0.98, 0.95),
            "ground": (0.90, 0.96, 0.92),
            "retailer": (0.30, 0.70, 0.55),
            "community": (0.98, 0.75, 0.50),
            "truck": (0.98, 0.50, 0.50),
            "route": (0.99, 0.62, 0.35),
        }

        self.camera_distance = 60.0
        self.camera_orbit = 0.0
        self.camera_pitch = 0.08

        self.supplies: List[float] = []
        self.demands: List[float] = []
        self.retailer_positions: List[np.ndarray] = []
        self.community_positions: List[np.ndarray] = []
        self.last_action: Optional[int] = None
        self.last_info: Dict[str, float] = {}
        self.max_supply = 1.0
        self.max_demand = 1.0

        self.animation = RouteAnimation()
        self._truck_pos = np.array([-20, 0, 4], dtype=np.float32)

        # Episode-level stats for HUD
        self.initial_supply: float = 0.0
        self.total_delivered: float = 0.0

    # ------------------------- public API ---------------------------------
    def update_state(self, env) -> None:
        """Pull state from the Gym environment."""
        self.supplies = list(getattr(env, "supplies", []))
        self.demands = list(getattr(env, "demands", []))
        self.last_info = getattr(env, "last_info", {}) or {}
        self.last_action = getattr(env, "last_action", None)
        self.max_supply = max(1.0, float(env.config.max_supply))
        self.max_demand = max(1.0, float(env.config.max_demand))

        time_step = getattr(env, "time_step", 0)
        if time_step == 0:
            self.initial_supply = float(np.sum(self.supplies))

        delivered_totals = getattr(env, "delivered_totals", [])
        self.total_delivered = float(np.sum(delivered_totals)) if len(delivered_totals) > 0 else 0.0

        base_y = [-10.0, 0.0, 10.0]
        left_x, right_x = -18.0, 18.0
        self.retailer_positions = [
            np.array([left_x, base_y[i], 4.0], dtype=np.float32)
            for i in range(min(len(self.supplies), len(base_y)))
        ]
        self.community_positions = [
            np.array([right_x, base_y[i], 4.0], dtype=np.float32)
            for i in range(min(len(self.demands), len(base_y)))
        ]

        if self.last_action is not None and self.community_positions:
            num_comm = len(self.community_positions)
            r_idx, c_idx = divmod(self.last_action, max(1, num_comm))
            if r_idx < len(self.retailer_positions) and c_idx < len(self.community_positions):
                self.animation = RouteAnimation(
                    active=True,
                    start_time=time.perf_counter(),
                    duration=1.1,
                    start=self.retailer_positions[r_idx],
                    end=self.community_positions[c_idx],
                )

    def adjust_zoom(self, delta: float) -> None:
        """Mouse wheel zoom."""
        self.camera_distance = float(np.clip(self.camera_distance - delta * 3.0, 25.0, 90.0))

    def adjust_orbit(self, dx: float, dy: float) -> None:
        """Mouse drag orbit."""
        self.camera_orbit += dx * 0.01
        self.camera_pitch = float(np.clip(self.camera_pitch + dy * 0.005, -0.1, 0.5))

    def render_frame(self) -> np.ndarray:
        """Render to framebuffer and return (H, W, 3) RGB array."""
        self.fbo.use()
        self.ctx.clear(*self.palette["bg"], 1.0)

        aspect = self.width / self.height
        proj = perspective(np.radians(46.0), aspect, 1.0, 300.0)
        eye = np.array(
            [
                self.camera_distance * np.sin(self.camera_orbit),
                -self.camera_distance * np.cos(self.camera_orbit),
                15.0 + self.camera_pitch * 40,
            ],
            dtype=np.float32,
        )
        view = look_at(eye, np.array([0, 0, 0], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32))

        self._draw_ground(view, proj)
        self._draw_nodes(view, proj)
        self._draw_roads(view, proj)
        self._draw_truck(view, proj)
        self._draw_highlight_route(view, proj)

        raw = self.fbo.read(components=3, alignment=1)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        frame = np.flipud(frame)
        frame = self._overlay_labels(frame, view, proj)
        return frame

    # ------------------------- scene drawing ---------------------------------
    def _draw_ground(self, view: np.ndarray, proj: np.ndarray) -> None:
        model = scale_matrix(30, 18, 1) @ translation_matrix(0, 0, -2.8)
        self.color_prog["mvp"].write((proj @ view @ model).T.astype(np.float32).tobytes())
        self.color_prog["color"].value = self.palette["ground"]
        self.cube_vao.render()

    def _draw_nodes(self, view: np.ndarray, proj: np.ndarray) -> None:
        for idx, pos in enumerate(self.retailer_positions):
            supply = self.supplies[idx] if idx < len(self.supplies) else 0.0
            height = 1.2 + 0.8 * (supply / self.max_supply)
            model = translation_matrix(*pos) @ scale_matrix(1.8, 1.8, height)
            self.color_prog["mvp"].write((proj @ view @ model).T.astype(np.float32).tobytes())
            self.color_prog["color"].value = self.palette["retailer"]
            self.cube_vao.render()

        for idx, pos in enumerate(self.community_positions):
            demand = self.demands[idx] if idx < len(self.demands) else 0.0
            height = 1.2 + 0.8 * (demand / self.max_demand)
            model = translation_matrix(*pos) @ scale_matrix(1.8, 1.8, height)
            self.color_prog["mvp"].write((proj @ view @ model).T.astype(np.float32).tobytes())
            self.color_prog["color"].value = self.palette["community"]
            self.cylinder_vao.render()

    def _draw_roads(self, view: np.ndarray, proj: np.ndarray) -> None:
        return

    def _draw_truck(self, view: np.ndarray, proj: np.ndarray) -> None:
        if self.animation.active:
            elapsed = time.perf_counter() - self.animation.start_time
            t = min(1.0, elapsed / self.animation.duration)
            control = (self.animation.start + self.animation.end) / 2 + np.array([0, 0, 10], dtype=np.float32)
            pos = (
                (1 - t) ** 2 * self.animation.start
                + 2 * (1 - t) * t * control
                + t ** 2 * self.animation.end
            )
            self._truck_pos = pos
            if t >= 1.0:
                self.animation.active = False

        model = translation_matrix(*self._truck_pos) @ scale_matrix(1.5, 3.0, 1.2)
        self.color_prog["mvp"].write((proj @ view @ model).T.astype(np.float32).tobytes())
        self.color_prog["color"].value = self.palette["truck"]
        self.cube_vao.render()

    def _draw_highlight_route(self, view: np.ndarray, proj: np.ndarray) -> None:
        if self.last_action is None:
            return
        num_comm = max(1, len(self.community_positions))
        r_idx, c_idx = divmod(self.last_action, num_comm)
        if r_idx >= len(self.retailer_positions) or c_idx >= len(self.community_positions):
            return
        start = self.retailer_positions[r_idx]
        end = self.community_positions[c_idx]
        control = (start + end) / 2 + np.array([0, 0, 8], dtype=np.float32)
        segments = 40
        points = []
        for i in range(segments + 1):
            t = i / segments
            points.append((1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end)
        vertices = np.vstack(points).astype(np.float32)
        vbo = self.ctx.buffer(vertices.tobytes())
        vao = self.ctx.simple_vertex_array(self.line_prog, vbo, "in_position")
        self.line_prog["mvp"].write((proj @ view).T.astype(np.float32).tobytes())
        self.line_prog["color"].value = self.palette["route"]
        vao.render(moderngl.LINE_STRIP)
        vbo.release()
        vao.release()

    def _overlay_labels(self, frame: np.ndarray, view: np.ndarray, proj: np.ndarray) -> np.ndarray:
        """Project supply/demand text into screen space using Pillow."""
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)

        def project(point: np.ndarray) -> Optional[Tuple[int, int]]:
            vec = np.append(point, 1.0)
            clip = proj @ view @ vec
            if clip[3] == 0:
                return None
            ndc = clip[:3] / clip[3]
            if not (-1 <= ndc[0] <= 1 and -1 <= ndc[1] <= 1 and -1 <= ndc[2] <= 1):
                return None
            x = int((ndc[0] * 0.5 + 0.5) * self.width)
            y = int((1 - (ndc[1] * 0.5 + 0.5)) * self.height)
            return x, y

        for idx, pos in enumerate(self.retailer_positions):
            screen = project(pos + np.array([0, 0, 6], dtype=np.float32))
            if screen:
                label = f"S:{self.supplies[idx]:.0f}" if idx < len(self.supplies) else "S:0"
                box = [screen[0] - 30, screen[1] - 14, screen[0] + 30, screen[1] + 10]
                draw.rounded_rectangle(box, radius=6, fill=(70, 170, 140, 220))
                draw.text((box[0] + 8, box[1] + 2), label, font=self.font, fill=(255, 255, 255))

        for idx, pos in enumerate(self.community_positions):
            screen = project(pos + np.array([0, 0, 6], dtype=np.float32))
            if screen:
                label = f"D:{self.demands[idx]:.0f}" if idx < len(self.demands) else "D:0"
                box = [screen[0] - 30, screen[1] - 14, screen[0] + 30, screen[1] + 10]
                draw.rounded_rectangle(box, radius=6, fill=(255, 210, 120, 220))
                draw.text((box[0] + 8, box[1] + 2), label, font=self.font, fill=(40, 40, 40))

        # Global HUD card with episode stats
        hud_margin_x = 20
        hud_margin_y = 20
        hud_width = 260
        hud_height = 70
        hud_box = [
            hud_margin_x,
            hud_margin_y,
            hud_margin_x + hud_width,
            hud_margin_y + hud_height,
        ]
        draw.rounded_rectangle(
            hud_box,
            radius=10,
            fill=(255, 255, 255, 230),
            outline=(180, 200, 190, 255),
            width=1,
        )
        line1 = f"Initial supply: {self.initial_supply:.1f} units"
        line2 = f"Total delivered: {self.total_delivered:.1f} units"
        text_x = hud_box[0] + 12
        text_y = hud_box[1] + 10
        draw.text((text_x, text_y), line1, font=self.font, fill=(30, 60, 60))
        draw.text((text_x, text_y + 24), line2, font=self.font, fill=(30, 60, 60))
        return np.array(image, dtype=np.uint8)

    # ------------------------- geometry creation ------------------------------
    def _create_cube(self) -> moderngl.VertexArray:
        positions = np.array(
            [
                # fmt: off
                -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1, -1,  1,   1,  1,  1,  -1,  1,  1,
                -1, -1, -1,  -1,  1, -1,   1,  1, -1,  -1, -1, -1,   1,  1, -1,   1, -1, -1,
                -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1, -1, -1,  -1,  1,  1,  -1,  1, -1,
                 1, -1, -1,   1,  1, -1,   1,  1,  1,   1, -1, -1,   1,  1,  1,   1, -1,  1,
                -1,  1, -1,  -1,  1,  1,   1,  1,  1,  -1,  1, -1,   1,  1,  1,   1,  1, -1,
                -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1, -1,   1, -1,  1,  -1, -1,  1,
                # fmt: on
            ],
            dtype=np.float32,
        )
        vbo = self.ctx.buffer(positions.tobytes())
        return self.ctx.simple_vertex_array(self.color_prog, vbo, "in_position")

    def _create_cylinder(self, segments: int = 32) -> moderngl.VertexArray:
        verts: List[float] = []
        for i in range(segments):
            theta0 = 2 * np.pi * i / segments
            theta1 = 2 * np.pi * (i + 1) / segments
            x0, y0 = np.cos(theta0), np.sin(theta0)
            x1, y1 = np.cos(theta1), np.sin(theta1)
            verts.extend([x0, y0, -1, x1, y1, -1, x1, y1, 1])
            verts.extend([x0, y0, -1, x1, y1, 1, x0, y0, 1])
        vertices = np.array(verts, dtype=np.float32)
        vbo = self.ctx.buffer(vertices.tobytes())
        return self.ctx.simple_vertex_array(self.color_prog, vbo, "in_position")

    def _create_billboard(self) -> moderngl.VertexArray:
        quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float32)
        vbo = self.ctx.buffer(quad.tobytes())
        return self.ctx.simple_vertex_array(self.billboard_prog, vbo, "in_pos")


