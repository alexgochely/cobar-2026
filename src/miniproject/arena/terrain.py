import numpy as np
from scipy.interpolate import RectBivariateSpline
from flygym.compose.world import FlatGroundWorld


def gaussian_blur_fft(height_map: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur in the frequency domain for smooth rolling terrain."""
    if sigma <= 0:
        return height_map

    ny, nx = height_map.shape
    fx = np.fft.fftfreq(nx)
    fy = np.fft.fftfreq(ny)
    fxx, fyy = np.meshgrid(fx, fy)

    # Frequency-domain Gaussian corresponding to spatial-domain blur.
    gaussian = np.exp(-2.0 * (np.pi**2) * (sigma**2) * (fxx**2 + fyy**2))
    filtered = np.fft.ifft2(np.fft.fft2(height_map) * gaussian).real
    return filtered


def generate_height_map(
    size_x: int,
    size_y: int,
    rng: np.random.Generator,
    blur_sigma: float,
) -> np.ndarray:
    base_noise = rng.standard_normal((size_y, size_x)).astype(np.float64)
    smooth = gaussian_blur_fft(base_noise, sigma=blur_sigma)

    smooth -= smooth.min()
    max_val = smooth.max()
    if max_val > 0:
        smooth /= max_val

    return smooth


class RollingHills(FlatGroundWorld):
    def __init__(
        self,
        name="rugged_terrain",
        pos=(0, 0, 0),
        s=256,
        xy_scale=0.5,
        amplitude=6.0,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(name=name)
        for geom in self.ground_contact_geoms:
            geom.remove()
        self.ground_contact_geoms.clear()

        height_map = generate_height_map(
            size_x=s,
            size_y=s,
            rng=rng or np.random.default_rng(seed=0),
            blur_sigma=8.0,
        )
        self.height_map = height_map
        self.hfield_pos = np.asarray(pos, dtype=np.float64)
        self.xy_scale = float(xy_scale)
        self.amplitude = float(amplitude)
        self.s = int(s)

        # Heightfield gives non-convex terrain collision, unlike mesh geoms.
        x_radius = (s - 1) * xy_scale / 2
        y_radius = (s - 1) * xy_scale / 2
        self.x_radius = float(x_radius)
        self.y_radius = float(y_radius)

        self._x_coords = np.linspace(-self.x_radius, self.x_radius, s, dtype=np.float64)
        self._y_coords = np.linspace(-self.y_radius, self.y_radius, s, dtype=np.float64)
        self._z_grid = self.height_map * self.amplitude + self.hfield_pos[2]
        kx = min(3, len(self._x_coords) - 1)
        ky = min(3, len(self._y_coords) - 1)
        # RectBivariateSpline expects z[i, j] = f(x[i], y[j]).
        self._surface = RectBivariateSpline(
            self._x_coords,
            self._y_coords,
            self._z_grid.T,
            kx=kx,
            ky=ky,
            s=0.0,
        )

        elevation = " ".join(map(str, height_map[::-1].ravel()))
        hfield = self.mjcf_root.asset.add(
            "hfield",
            name="rugged_hfield",
            nrow=s,
            ncol=s,
            elevation=elevation,
            size=(x_radius, y_radius, amplitude, 1),
        )
        ground_geom = self.mjcf_root.worldbody.add(
            "geom",
            type="hfield",
            hfield=hfield,
            name="ground_plane",
            pos=pos,
            contype=0,
            conaffinity=0,
            rgba=(0, 1, 0, 1),
        )
        self.ground_contact_geoms.append(ground_geom)

    def get_height(self, x: float, y: float) -> float:
        """Return terrain height z at world coordinates (x, y)."""
        local_x = float(x) - self.hfield_pos[0]
        local_y = float(y) - self.hfield_pos[1]
        local_x = float(np.clip(local_x, self._x_coords[0], self._x_coords[-1]))
        local_y = float(np.clip(local_y, self._y_coords[0], self._y_coords[-1]))
        return float(self._surface.ev(local_x, local_y))

    def get_normal(self, x: float, y: float, eps: float | None = None) -> np.ndarray:
        """Return an upward unit normal vector at world coordinates (x, y)."""
        local_x = float(x) - self.hfield_pos[0]
        local_y = float(y) - self.hfield_pos[1]
        local_x = float(np.clip(local_x, self._x_coords[0], self._x_coords[-1]))
        local_y = float(np.clip(local_y, self._y_coords[0], self._y_coords[-1]))

        dz_dx = float(self._surface.ev(local_x, local_y, dx=1, dy=0))
        dz_dy = float(self._surface.ev(local_x, local_y, dx=0, dy=1))

        normal = np.array([-dz_dx, -dz_dy, 1.0], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normal / norm
