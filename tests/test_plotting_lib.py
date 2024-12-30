from unittest import TestCase
import numpy as np
import mpmath as mp
import plotting_lib

class Test(TestCase):
    def test_to_mpf(self):
        mp.mp.dps = 50
        self.assertEqual(mp.mpf("0.1"), plotting_lib._to_mpf(0.1))
        self.assertEqual(mp.mpf("0.135123665221742145526159"),
                         plotting_lib._to_mpf(mp.mpf("0.135123665221742145526159")))
        self.assertEqual(mp.mpf("0.135123665221742145526159"),
                         plotting_lib._to_mpf("0.135123665221742145526159"))

    def test_get_ticks(self):
        scaled_ticks, ticks = plotting_lib._get_ticks(1.04, 1.05, 3, 1.04, 1.05)
        self.assertTrue(np.abs(np.array([mp.mpf(0), mp.mpf(0.5), mp.mpf(1)]) - np.array(scaled_ticks)).max() < 1e-13)
        self.assertEqual(["1.04", "1.045", "1.05"], ticks)
        scaled_ticks, ticks = plotting_lib._get_ticks("1.04", "1.05", 3, 1.04, 1.05)
        self.assertTrue(np.abs(np.array([mp.mpf(0), mp.mpf(0.5), mp.mpf(1)]) - np.array(scaled_ticks)).max() < 1e-13)
        self.assertEqual(["1.04", "1.045", "1.05"], ticks)
        scaled_ticks, ticks = plotting_lib._get_ticks("1.04", "1.05", 3, mp.mpf("1.04"), mp.mpf("1.05"))
        self.assertTrue(np.abs(np.array([mp.mpf(0), mp.mpf(0.5), mp.mpf(1)]) - np.array(scaled_ticks)).max() < 1e-13)
        self.assertEqual(["1.04", "1.045", "1.05"], ticks)
        mp.mp.dps = 50
        scaled_ticks, ticks = plotting_lib._get_ticks("1.040450456783456454564357", "1.040450456783456454564358", 3,
                                                      mp.mpf("1.040450456783456454564357"),
                                                      mp.mpf("1.040450456783456454564358"))
        self.assertTrue(np.abs(np.array([mp.mpf(0), mp.mpf(0.5), mp.mpf(1)]) - np.array(scaled_ticks)).max() < 1e-13)
        self.assertEqual(["1.040450456783456454564357", "1.0404504567834564545643575", "1.040450456783456454564358"],
                         ticks)
        scaled_ticks, ticks = plotting_lib._get_ticks("1.040450456783456454564357", "1.040450456783456454564358", 3,
                                                      mp.mpf("1.040450456783456454564355"),
                                                      mp.mpf("1.040450456783456454564359"))
        self.assertTrue(np.abs(np.array([mp.mpf(0.5), mp.mpf(0.625), mp.mpf(0.75)]) - np.array(scaled_ticks)).max()
                        < 1e-13)
        self.assertEqual(["1.040450456783456454564357", "1.0404504567834564545643575", "1.040450456783456454564358"],
                         ticks)
        scaled_ticks, ticks = plotting_lib._get_ticks("-0.2", "1", 7, 0, 1)
        self.assertEqual(["-0.2", "0", "0.2", "0.4", "0.6", "0.8", "1.0"], ticks)
        scaled_ticks, ticks = plotting_lib._get_ticks("-0.1", "0.11", 7, 0, 1)
        self.assertEqual(["-0.1", "-0.065", "-0.03", "0.005", "0.04", "0.075", "0.11"], ticks)

    def test_get_based_ticks(self):
        scaled_ticks, ticks = plotting_lib._get_based_ticks(1.04, 1.05, 0, 0.005)
        np.testing.assert_array_almost_equal(scaled_ticks, np.linspace(0, 1, 3), decimal=14)
        self.assertEqual(ticks, ["1.04", "1.045", "1.05"])
        scaled_ticks, ticks = plotting_lib._get_based_ticks(1.04, 1.05, 1, 0.005)
        np.testing.assert_array_almost_equal(scaled_ticks, np.linspace(0, 1, 3), decimal=14)
        self.assertEqual(ticks, ["0.04", "0.045", "0.05"])
        mp.mp.dps = 50
        scaled_ticks, ticks = plotting_lib._get_based_ticks(
            "1.040450456783456454564357", "1.040450456783456454564358", 0, 5e-25)
        np.testing.assert_array_almost_equal(scaled_ticks, np.linspace(0, 1, 3), decimal=14)
        self.assertEqual(
            ["1.040450456783456454564357", "1.0404504567834564545643575", "1.040450456783456454564358"],
            ticks)
        scaled_ticks, ticks = plotting_lib._get_based_ticks(
            "1.040450456783456454564357", "1.040450456783456454564358",
            "1.040450456783456454564358", 5e-25)
        np.testing.assert_array_almost_equal(scaled_ticks, np.linspace(0, 1, 3), decimal=14)
        self.assertEqual(["-1.0e-24", "-5.0e-25", "0.0"], ticks)

