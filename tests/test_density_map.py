from __future__ import annotations

import unittest

from src.utils.density_map import compute_adaptive_sigmas, make_density_map


class DensityMapTest(unittest.TestCase):
    def test_compute_adaptive_sigmas_uses_nearest_neighbor_distance(self) -> None:
        points = [(0.0, 0.0), (10.0, 0.0), (100.0, 0.0)]

        sigmas = compute_adaptive_sigmas(points, beta=0.3, min_sigma=1.0, max_sigma=32.0)

        self.assertEqual(sigmas, [3.0, 3.0, 27.0])

    def test_adaptive_density_map_preserves_point_count(self) -> None:
        density = make_density_map(
            [(16.0, 16.0), (48.0, 48.0), (90.0, 90.0)],
            height=128,
            width=128,
            sigma=4.0,
            downsample=8,
            sigma_mode="adaptive",
            adaptive_sigma_beta=0.3,
        )

        self.assertEqual(density.shape, (16, 16))
        self.assertAlmostEqual(float(density.sum()), 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
