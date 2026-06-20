import math
import numpy as np
import taichi as ti


def enu_to_ti(pts: np.ndarray) -> np.ndarray:
    """ENU (x-East, y-North, z-Up) -> Taichi (x, y=z_enu, z=-y_enu)"""
    out = pts[..., [0, 2, 1]].copy()
    out[..., 2] *= -1
    return out


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: [w, x, y, z] -> R: (3, 3)"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


class BaseRenderer:
    """
    Live single-step renderer. Call step(state) once per env.step().

    state: np.ndarray (13,)
    [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]

    Returns False if the user pressed X/Esc (signal to stop simulation).

    Subclasses implement:
      _draw_extras(scene)
      _draw_hud(window)
      _handle_keys(window)
    """

    _AXIS_COLORS = [
        (0.9, 0.15, 0.15),
        (0.15, 0.9, 0.15),
        (0.15, 0.15, 0.9),
    ]

    def __init__(
        self,
        arm_length:  float = 0.15,
        arm_angle:   float = 45.0,
        mass:        float = 1.0,
        dt:          float = 0.01,
        window_size: tuple = (1280, 720),
    ):
        self._dt        = dt
        self._arm_len   = arm_length
        self._axis_scale = arm_length * 1.5
        self._sphere_r  = arm_length * 0.08 * (mass / 0.5) ** (1/3)
        self._motor_r   = self._sphere_r * 0.75

        self._paused      = False
        self._show_ground = False

        s = math.sin(math.radians(arm_angle))
        c = math.cos(math.radians(arm_angle))
        self._arm_dirs_body = arm_length * np.array([
            [ c, -s, 0],
            [-c,  s, 0],
            [ c,  s, 0],
            [-c, -s, 0],
        ], dtype=np.float32)                        # (4, 3)

        ti.init(arch=ti.cpu)

        # Taichi fields — written each step
        self._body_pos        = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self._arm_verts       = ti.Vector.field(3, dtype=ti.f32, shape=8)
        self._motor_pos       = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self._body_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)
        ]

        self._build_ground()
        self._build_world_axes()
        self._window_size = window_size
        self._window = None
        self._canvas = None
        self._scene = None
        self._camera = ti.ui.Camera()
        self._camera.position(0, 3, 5)
        self._camera.lookat(0, 0, 0)
        self._camera.up(0, 1, 0)

        self._step_count = 0

    # ── Ground ────────────────────────────────────────────────────────────

    def _build_ground(self):
        grid_n    = 10
        grid_half = 5
        edge      = 2.0 * grid_half / (grid_n - 1)

        self._grid_verts = ti.Vector.field(3, dtype=ti.f32, shape=4 * grid_n)
        gi = 0
        for k in range(grid_n):
            v = -grid_half + 2 * grid_half * k / (grid_n - 1)
            self._grid_verts[gi+0] = ti.Vector([v,          0.002, -grid_half])
            self._grid_verts[gi+1] = ti.Vector([v,          0.002,  grid_half])
            self._grid_verts[gi+2] = ti.Vector([-grid_half, 0.002,  v])
            self._grid_verts[gi+3] = ti.Vector([ grid_half, 0.002,  v])
            gi += 4

        num_cells      = (grid_n - 1) ** 2
        self._ground_v = ti.Vector.field(3, dtype=ti.f32, shape=num_cells * 4)
        self._ground_c = ti.Vector.field(3, dtype=ti.f32, shape=num_cells * 4)
        self._ground_i = ti.field(dtype=ti.i32,           shape=num_cells * 6)

        dark  = ti.Vector([0.125, 0.239, 0.322])
        light = ti.Vector([0.243, 0.439, 0.592])

        for i in range(grid_n - 1):
            for j in range(grid_n - 1):
                cell = i * (grid_n - 1) + j
                vb   = cell * 4
                x0 = -grid_half + i * edge;  x1 = x0 + edge
                z0 = -grid_half + j * edge;  z1 = z0 + edge

                self._ground_v[vb+0] = ti.Vector([x0, 0.0, z0])
                self._ground_v[vb+1] = ti.Vector([x1, 0.0, z0])
                self._ground_v[vb+2] = ti.Vector([x0, 0.0, z1])
                self._ground_v[vb+3] = ti.Vector([x1, 0.0, z1])

                col = dark if (i + j) % 2 == 0 else light
                for k in range(4):
                    self._ground_c[vb+k] = col

                ib = cell * 6
                self._ground_i[ib+0] = vb;     self._ground_i[ib+1] = vb+1
                self._ground_i[ib+2] = vb+2;   self._ground_i[ib+3] = vb+1
                self._ground_i[ib+4] = vb+3;   self._ground_i[ib+5] = vb+2

    # ── World axes ────────────────────────────────────────────────────────

    def _build_world_axes(self):
        s = self._axis_scale
        tips = [[s, 0, 0], [0, s, 0], [0, 0, -s]]
        self._world_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=2) for _ in range(3)
        ]
        for i in range(3):
            self._world_axis_verts[i][0] = ti.Vector([0.0, 0.0, 0.0])
            self._world_axis_verts[i][1] = ti.Vector(tips[i])

    # ── Extension points ──────────────────────────────────────────────────

    def _draw_extras(self, scene):
        pass

    def _draw_hud(self, window):
        with window.GUI.sub_window("Info", 0.01, 0.01, 0.30, 0.08) as sw:
            sw.text(f"Step: {self._step_count}   Time: {self._step_count * self._dt:.2f}s")

    def _handle_keys(self, window):
        pass

    def _ensure_window(self):
        if self._window is None:
            self._window = ti.ui.Window("Quadrotor", self._window_size, vsync=True)
            self._canvas = self._window.get_canvas()
            self._scene = self._window.get_scene()

    # ── Main API ──────────────────────────────────────────────────────────

    def step(self, state: np.ndarray) -> bool:
        """
        Update and render one frame.
        Returns False if the user pressed X/Esc → env should stop.
        """
        self._ensure_window()

        if not self._window.running:
            return False

        # Key events
        if self._window.get_event(ti.ui.PRESS):
            if self._window.event.key in (ti.ui.ESCAPE, 'x'):
                return False
            if self._window.event.key == ' ':
                self._paused = not self._paused
            if self._window.event.key == 'g':
                self._show_ground = not self._show_ground
            self._handle_keys(self._window)        

        # Update Taichi fields from state
        p   = state[0:3]
        q   = state[6:10]
        R   = quat_to_rotmat(q)
        p_ti = enu_to_ti(p[None])[0]
        origin = ti.Vector(p_ti.tolist())

        self._body_pos[0] = origin

        bases_enu = self._axis_scale * np.eye(3, dtype=np.float32)
        tips_enu  = p + (R @ bases_enu.T).T                        # (3, 3)
        tips_ti   = enu_to_ti(tips_enu)
        for i in range(3):
            self._body_axis_verts[i][0] = origin
            self._body_axis_verts[i][1] = ti.Vector(tips_ti[i].tolist())

        arm_tips_enu = p + (R @ self._arm_dirs_body.T).T           # (4, 3)
        arm_tips_ti  = enu_to_ti(arm_tips_enu)
        for i in range(4):
            tip = ti.Vector(arm_tips_ti[i].tolist())
            self._arm_verts[i*2]   = origin
            self._arm_verts[i*2+1] = tip
            self._motor_pos[i]     = tip

        # Draw
        self._camera.track_user_inputs(self._window, movement_speed=0.05, hold_key=ti.ui.LMB)
        self._scene.set_camera(self._camera)
        self._scene.ambient_light((0.3, 0.3, 0.3))
        self._scene.point_light((0.0, 10.0,  0.0), color=(1.0, 1.0, 1.0))
        self._scene.point_light((-10.0, 10.0, 0.0), color=(1.0, 1.0, 1.0))

        if self._show_ground:
            self._scene.mesh(self._ground_v, indices=self._ground_i, per_vertex_color=self._ground_c)
        self._scene.lines(self._grid_verts, width=1.0, color=(0.2, 0.2, 0.3))

        for i in range(3):
            self._scene.lines(self._world_axis_verts[i], width=3.0, color=self._AXIS_COLORS[i])

        self._scene.particles(self._body_pos,  radius=self._sphere_r, color=(0.2, 0.2, 0.2))
        self._scene.lines(    self._arm_verts,  width=3.0,            color=(0.85, 0.85, 0.85))
        self._scene.particles(self._motor_pos,  radius=self._motor_r, color=(0.1, 0.6, 0.9))

        for i in range(3):
            self._scene.lines(self._body_axis_verts[i], width=2.0, color=self._AXIS_COLORS[i])

        self._draw_extras(self._scene)

        self._canvas.scene(self._scene)
        self._draw_hud(self._window)
        self._window.show()

        if self._paused:
            return True 

        self._step_count += 1
        return True
    
    @property
    def is_paused(self):
        return self._paused

    def close(self):
        if self._window is not None:
            self._window.destroy()
            self._window = None
            self._canvas = None
            self._scene = None


class PosCtrlRenderer(BaseRenderer):
    """
    Extends BaseRenderer with:
      - A target sphere at a given position
      - A boundary wireframe cube (toggle with 'b')
    """

    def __init__(
        self,
        target:      np.ndarray,        # (3,) ENU target position
        bounds:      tuple = (3.5, 2.5, 2.5),  # (x, y, z) half-extents
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._show_bounds = True

        # Target spheres
        target_ti = enu_to_ti(np.array(target, dtype=np.float32)[None])[0]
        self._target_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)
        self._target_pos[0] = ti.Vector(target_ti.tolist())

        # Boundary box wireframe — 12 edges × 2 verts
        self._bounds_verts = ti.Vector.field(3, dtype=ti.f32, shape=24)
        bx, by, bz = bounds
        # 8 corners in ENU, converted to Taichi
        corners_enu = np.array([
            [-bx, -by, -bz], [ bx, -by, -bz],
            [ bx,  by, -bz], [-bx,  by, -bz],
            [-bx, -by,  bz], [ bx, -by,  bz],
            [ bx,  by,  bz], [-bx,  by,  bz],
        ], dtype=np.float32)
        c = enu_to_ti(corners_enu)  # (8, 3)

        edges = [
            (0,1),(1,2),(2,3),(3,0),  # bottom face
            (4,5),(5,6),(6,7),(7,4),  # top face
            (0,4),(1,5),(2,6),(3,7),  # verticals
        ]
        for i, (a, b) in enumerate(edges):
            self._bounds_verts[i*2]   = ti.Vector(c[a].tolist())
            self._bounds_verts[i*2+1] = ti.Vector(c[b].tolist())

    def _handle_keys(self, window):
        if window.event.key == 'b':
            self._show_bounds = not self._show_bounds

    def _draw_extras(self, scene):
        scene.particles(self._target_pos, radius=self._sphere_r * 2.0, color=(0.2, 1.0, 0.2))
        if self._show_bounds:
            scene.lines(self._bounds_verts, width=2.0, color=(0.8, 0.2, 0.2))

    def set_target(self, target: np.ndarray):
        """target: (3,) ENU"""
        target_ti = enu_to_ti(target[None])[0]
        self._target_pos[0] = ti.Vector(target_ti.tolist())


class RacingRenderer(BaseRenderer):
    """
    Extends BaseRenderer with racing gates loaded from positions/rpy arrays.
    Active gate is highlighted with a different color.
    """

    def __init__(
        self,
        gates_position:       np.ndarray,
        gates_rpy:            np.ndarray,
        gate_mesh_path:       str   = "/home/adame/AirBender/misc/gate.obj",
        gate_scale:           float = 1.0,
        gate_color:           tuple = (0.25, 0.0, 0.5),
        active_gate_color:    tuple = (0.8, 0.8, 0.0),
        gate_mesh_rpy_offset: tuple = (90.0, 0.0, 0.0),
        **kwargs,
    ):
        super().__init__(**kwargs)

        try:
            import trimesh
        except ImportError:
            raise ImportError("pip install trimesh")

        raw = trimesh.load(gate_mesh_path, force="mesh")
        if isinstance(raw, trimesh.Scene):
            raw = trimesh.util.concatenate(tuple(raw.geometry.values()))

        verts_body = raw.vertices.astype(np.float32) * gate_scale
        faces      = raw.faces.astype(np.int32)
        V, F       = len(verts_body), len(faces)
        N          = len(gates_position)

        rpy_offset = np.array(gate_mesh_rpy_offset, dtype=np.float32)
        all_verts_enu = np.zeros((N * V, 3), dtype=np.float32)
        for g in range(N):
            R = self._rpy_to_rotmat(gates_rpy[g] + rpy_offset)
            all_verts_enu[g*V:(g+1)*V] = (R @ verts_body.T).T + gates_position[g]

        all_verts_ti = enu_to_ti(all_verts_enu)
        all_faces    = np.zeros((N * F, 3), dtype=np.int32)
        for g in range(N):
            all_faces[g*F:(g+1)*F] = faces + g * V

        self._gate_color        = gate_color
        self._active_gate_color = active_gate_color
        self._V, self._F, self._N = V, F, N
        self._all_verts_ti = all_verts_ti
        self._all_faces    = all_faces

        self._has_inactive = N > 1

        self._active_v   = ti.Vector.field(3, dtype=ti.f32, shape=V)
        self._active_i   = ti.field(dtype=ti.i32,           shape=F * 3)
        if self._has_inactive:
            self._inactive_v = ti.Vector.field(3, dtype=ti.f32, shape=(N - 1) * V)
            self._inactive_i = ti.field(dtype=ti.i32,           shape=(N - 1) * F * 3)
        else:
            self._inactive_v = None
            self._inactive_i = None

        self._upload_gate_split(0)

        # Gate axes — semantic orientation
        axis_scale = gate_scale * 0.4
        axis_dirs  = np.eye(3, dtype=np.float32) * axis_scale

        self._gate_axis_verts = [
            ti.Vector.field(3, dtype=ti.f32, shape=N * 2) for _ in range(3)
        ]

        for g in range(N):
            R          = self._rpy_to_rotmat(gates_rpy[g])     
            origin_enu = gates_position[g]
            tips_enu   = origin_enu + (R @ axis_dirs.T).T      # (3, 3)

            origin_ti  = enu_to_ti(origin_enu[None])[0]
            tips_ti    = enu_to_ti(tips_enu)

            for axis_id in range(3):
                self._gate_axis_verts[axis_id][g*2]   = ti.Vector(origin_ti.tolist())
                self._gate_axis_verts[axis_id][g*2+1] = ti.Vector(tips_ti[axis_id].tolist())

    # ------------------------------------------------------------------
    @staticmethod
    def _rpy_to_rotmat(rpy_deg: np.ndarray) -> np.ndarray:
        r, p, y = np.radians(rpy_deg)
        cr, sr  = math.cos(r), math.sin(r)
        cp, sp  = math.cos(p), math.sin(p)
        cy, sy  = math.cos(y), math.sin(y)
        Rx = np.array([[1,  0,   0], [0,  cr, -sr], [0,  sr, cr]], dtype=np.float32)
        Ry = np.array([[cp, 0,  sp], [0,   1,   0], [-sp, 0, cp]], dtype=np.float32)
        Rz = np.array([[cy, -sy, 0], [sy,  cy,  0], [0,   0,  1]], dtype=np.float32)
        return Rz @ Ry @ Rx

    # ------------------------------------------------------------------
    def _upload_gate_split(self, active_idx: int):
        V, F, N  = self._V, self._F, self._N
        all_v    = self._all_verts_ti
        all_f    = self._all_faces

        # Active gate
        self._active_v.from_numpy(all_v[active_idx*V : (active_idx+1)*V])
        self._active_i.from_numpy(
            (all_f[active_idx*F : (active_idx+1)*F] - active_idx*V).flatten()
        )

        if not self._has_inactive:
            return

        # Inactive gates
        inactive_verts = np.concatenate([
            all_v[g*V:(g+1)*V] for g in range(N) if g != active_idx
        ])
        inactive_faces = []
        new_idx = 0
        for g in range(N):
            if g == active_idx:
                continue
            inactive_faces.append(all_f[g*F:(g+1)*F] - g*V + new_idx*V)
            new_idx += 1

        self._inactive_v.from_numpy(inactive_verts)
        self._inactive_i.from_numpy(np.concatenate(inactive_faces).flatten())

    # ------------------------------------------------------------------
    def set_target(self, gate_idx: int):
        self._upload_gate_split(gate_idx)

    # ------------------------------------------------------------------
    def _draw_extras(self, scene):
        if self._has_inactive:
            scene.mesh(self._inactive_v, indices=self._inactive_i, color=self._gate_color)
        scene.mesh(self._active_v, indices=self._active_i, color=self._active_gate_color)
        for i in range(3):
            scene.lines(self._gate_axis_verts[i], width=2.0, color=self._AXIS_COLORS[i])