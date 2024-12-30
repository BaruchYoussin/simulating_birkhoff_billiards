import json
import os.path
import pathlib
import pickle
import shutil
from unittest import TestCase
import mpmath as mp
import numpy as np
import pandas as pd
import billiard_Birkhoff_lib
import general_lib


class Test(TestCase):
    def test_h_and_deriv(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        h, h_d = billiard.h_and_deriv(mp.mpf(0))
        self.assertAlmostEqual(mp.mpf("0.766044443118978035202392650555"), h, places=30)
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 500)
        h, h_d = billiard.h_and_deriv(mp.mpf(0.1))
        self.assertAlmostEqual(
            mp.mpf("0.771501629894024761448237990249932189671951635908134249073378493340885845056522754865786930130052856024364593854602276232085559680007144404960116445551153343890510599799897175057809882166710640618337780551864905227455823961667036771191173671597845184906219344639482963631521396927961326651479727882048452620983836380005799837829638701332067232620147248250822129562110686803424523089711565700550853073198916872845732410252954229293886996227490053931068964452365755589394752037894866689965627255818381868"),
            h, places=500)
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        h, h_d = billiard.h_and_deriv(mp.mpf(0))
        self.assertAlmostEqual(mp.mpf(0), h_d, places=50)
        h, h_d = billiard.h_and_deriv(mp.mpf("0.1"))
        self.assertAlmostEqual(mp.mpf("0.10700592670261433776390696522440040925946561175986"), h_d, places=50)
        # Here the value is Mathematica's SetPrecision[0.1, 50]:
        h, h_d = billiard.h_and_deriv(mp.mpf("0.10000000000000000555111512312578270211815834045410"))
        self.assertAlmostEqual(mp.mpf("0.10700592670261434324054862358167621440868460002851"), h_d, places=50)

    def test_theoretical_invariant_curve(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        phi, p = billiard.theoretical_invariant_curve(0)
        self.assertAlmostEqual(phi, mp.pi/9, places=29)
        self.assertAlmostEqual(p, mp.mpf(1/4) + mp.cos(mp.pi/9) / 2, places=29)
        phi, p = billiard.theoretical_invariant_curve(2 * mp.pi/3)
        self.assertAlmostEqual(phi, 7 * mp.pi/9, places=29)
        self.assertAlmostEqual(p, mp.mpf(1/4) + mp.cos(mp.pi/9) / 2, places=29)
        phi, p = billiard.theoretical_invariant_curve(mp.pi/6)
        self.assertAlmostEqual(phi, mp.pi / 3, places=29)
        self.assertAlmostEqual(p, mp.mpf(7 / 8), places=29)

    def test_filename(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        self.assertEqual("orbits/k3A0.5/orbit_k3A0.5phi0p0.7prec30n_iter1000.pkl",
                         billiard.filename(0, 0.7, 3, 1000))

    def test_dir_orbit_rectangle(self):
        phi, p = '1.0471975511965977459651997100036728260448414392093', '0.90669430721536456940936708034301662102564491748964'
        num_iter = 50 * 1000 * 1000
        rectangle = ["1.047197551196597745", "1.047197551196597747", "0.906694307215364569408", "0.906694307215364569412"]
        n_decimals_in_name_phi = 21
        n_decimals_in_name_p = 24
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 80)
        dir_path = billiard.dir_orbit_rectangle(phi, p, num_iter, rectangle, n_decimals_in_name_phi, n_decimals_in_name_p)
        expected = "orbits/k3A0.5/orbit_k3A0.5phi1.04719755119659774597p0.906694307215364569409367prec80n_iter50000000/phi1.047197551196597745to1.047197551196597747p0.906694307215364569408to0.906694307215364569412"
        self.assertEqual(dir_path, expected)
        phi, p = mp.mpf('1.0471975511965977459651997100036728260448414392093'), mp.mpf('0.90669430721536456940936708034301662102564491748964')
        dir_path = billiard.dir_orbit_rectangle(phi, p, num_iter, rectangle, n_decimals_in_name_phi, n_decimals_in_name_p)
        self.assertEqual(dir_path, expected)

    def test_param_boundary(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        param_boundary_fn = billiard.param_boundary()
        x, y = param_boundary_fn(0)
        self.assertAlmostEqual(x, mp.cos(2 * mp.pi / 9), places=29)
        self.assertAlmostEqual(y, 0, places=29)
        x, y = param_boundary_fn(mp.pi/6)
        self.assertAlmostEqual(x, 5 / 8, places=29)
        self.assertAlmostEqual(y, 3 * mp.sqrt(3) / 8, places=29)

    def test_compare_orbits(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        orbit1 = np.array([[1, mp.mpf("0.112345678901234567890123456789"), mp.mpf("0.345678901234567890123456789012")],
                           [2, mp.mpf("3.1415926535897932384626433"), mp.mpf("0.567890123456789012345678901234")]])
        orbit2 = np.array([[0, 1]])
        error_message = ""
        try:
            billiard.compare_orbits(orbit1, orbit2)
        except ValueError as e:
            error_message = str(e)
        self.assertEqual('Orbits that are compared, must be similar. Trying to compare orbits with '
                         'the shapes\n((2, 3), (1, 2)) and numbering ranging between ((1, 2)) and'
                         ' ((np.int64(0), np.int64(0)))',
                         error_message)
        orbit1_ext = np.expand_dims(orbit1, 2)
        orbit2_ext = np.expand_dims(orbit2, 2)
        error_message = ""
        try:
            billiard.compare_orbits(orbit1_ext, orbit2_ext)
        except ValueError as e:
            error_message = str(e)
        self.assertEqual('Orbits that are compared, must be similar. Trying to compare orbits with '
                         'the shapes\n((2, 3, 1), (1, 2, 1)) and numbering ranging between '
                         '((array([1], dtype=object), array([2], dtype=object))) and ((array([0]), array([0])))',
                         error_message)
        orbit2 = np.array([[0, 1, 1], [2, 3, 1]])
        try:
            billiard.compare_orbits(orbit1, orbit2)
        except ValueError as e:
            error_message = str(e)
        self.assertEqual('Orbits that are compared, must be similar. Trying to compare orbits with the '
                         'shapes\n((2, 3), (2, 3)) and numbering ranging between ((1, 2)) and '
                         '((np.int64(0), np.int64(2)))',
                         error_message)
        orbit2 = np.array([[1, mp.mpf("0"), mp.mpf("0")], [2, mp.mpf("3"), mp.mpf("0")]])
        actual = billiard.compare_orbits(orbit1, orbit2)
        expected = mp.mpf("0.567890123456789012345678901234")
        self.assertAlmostEqual(expected, actual, places=25)
        orbit2 = np.array([[1, mp.mpf("0.112345678901234567890123456789"), mp.mpf("0.345678901234567890123456789012")],
                           [2, mp.mpf("-3.1415926535897932384626433"), mp.mpf("0.567890123456789012345678901234")]])
        actual = billiard.compare_orbits(orbit1, orbit2)
        expected = mp.mpf('1.66558513806553990571021545816677e-25')
        self.assertEqual(expected, actual)
        orbit2_ext = np.expand_dims(orbit2, 2)
        actual = billiard.compare_orbits(orbit1_ext, orbit2_ext)
        self.assertEqual(expected, actual)
        orbit1_ext = np.stack([orbit1] * 2, 2)
        orbit2_ext = np.stack([orbit2] * 2, 2)
        actual = billiard.compare_orbits(orbit1_ext, orbit2_ext)
        self.assertEqual(expected, actual)
        orbit2_ext = np.stack((orbit1, orbit2), 2)
        actual = billiard.compare_orbits(orbit1_ext, orbit2_ext)
        self.assertEqual(expected, actual)
        orbit2_ext = np.stack([orbit2, orbit1], 2)
        actual = billiard.compare_orbits(orbit1_ext, orbit2_ext)
        self.assertEqual(expected, actual)


    def test_set_orbit_around_point(self):
        orbit = np.array([[0, 1, 2], [1, 5, 4]], dtype="object")
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        np.testing.assert_array_almost_equal(billiard.set_orbit_around_point(orbit, np.array([0, 0])),
                                             np.array([[0, 1, 2], [1, 5 - 2 * mp.pi, 4]]),
                                             decimal=49)

    def test_build_orbit_by_blocks(self):
        final_filepath = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.build_orbit_by_blocks(
            3, "0.5", 50, 70, 0.1, 0.6, 4, 2,
            [3, 5, 0.65, 1], 1, "../")
        actual1 = general_lib.unpickle(final_filepath)
        full_orbit1 = np.array(
            [[0, mp.mpf('0.10000000000000000555111512312578270211815834045410156'),
              mp.mpf('0.59999999999999997779553950749686919152736663818359375')],
             [1, mp.mpf('1.6120800624178759229699463278843131593259519408349936'),
              mp.mpf('0.75179379600467554117001778134832363269998336797650645')],
             [2, mp.mpf('2.3885575418758013530823756554971554447063111545136467'),
              mp.mpf('0.67523369701067162906524486253688838137369871578850557')],
             [3, mp.mpf('3.9800092720241094074161887899304061332100057787450261'),
              mp.mpf('0.63914190200217508334610692107607949045665683785813668')],
             [4, mp.mpf('4.8203530576808731353293115046360311355561032779515226'),
              mp.mpf('0.80141497901039523963131202216294403204310318208661886')]])
        np.testing.assert_array_almost_equal(actual1, full_orbit1.take([0, 4], axis=0), decimal=49)
        destination_dir = os.path.dirname(final_filepath)
        self.assertEqual(destination_dir, "../orbits/k3A0.5/orbit_k3A0.5phi0.1p0.6prec50n_iter4/phi3to5p0.65to1")
        shutil.rmtree(os.path.dirname(destination_dir))
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95F\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x01\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x08\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x01\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(G?\x9cd\xa0y\xc3\xcc\x84G?\x9cj0\xdc|\xff@NNNet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h8\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h:h<}\x94(h>h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x01\x85\x94h\x1b\x89C\x08\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhO\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      f"{destination_dir}/misc_outputs.pkl")
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\xf2\x01\x00\x00\x00\x00\x00\x00\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x03K\x03\x86\x94h\x03\x8c\x05dtype\x94\x93\x94\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(K\x00\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\rccccccccccccd\x94J\xc9\xff\xff\xffK4t\x94bh\x15)\x81\x94(K\x00\x8c\x0e13333333333333\x94J\xcb\xff\xff\xffK5t\x94bK\x01h\x15)\x81\x94(K\x00\x8c+19cb1476a9e507f08b48b0c60bcce876da5c79bab8d\x94JX\xff\xff\xffK\xa9t\x94bh\x15)\x81\x94(K\x00\x8c+180eb1dce59ff6e7470a1db4458959c369230e17ba1\x94JW\xff\xff\xffK\xa9t\x94bK\x02h\x15)\x81\x94(K\x00\x8c*98de2073be203b79f60ed7b9c04a1d35aba9344b81\x94JZ\xff\xff\xffK\xa8t\x94bh\x15)\x81\x94(K\x00\x8c*566e0ecae8b4792e0cd146de312c03883309138b85\x94JY\xff\xff\xffK\xa7t\x94bet\x94b.'),
                                      f"{destination_dir}/main_block_current.pkl")
        final_filepath = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.build_orbit_by_blocks(
            3, "0.5", 50, 70, 0.1, 0.6, 4, 2,
            [3, 5, 0.65, 1], 2, "../", 3)
        actual1 = general_lib.unpickle(final_filepath)
        np.testing.assert_array_almost_equal(actual1, full_orbit1.take([0, 4], axis=0), decimal=50)
        destination_dir = os.path.dirname(final_filepath)
        self.assertEqual(destination_dir, "../orbits/k3A0.5/orbit_k3A0.5phi0.1p0.6prec50n_iter4/phi3to5p0.65to1")
        shutil.rmtree(os.path.dirname(destination_dir))

    def test_extend_orbit_by_blocks(self):
        partial_orbit_path = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.build_orbit_by_blocks(
            3, "0.5", 50, 70, 0.1, 0.6, 2, 2,
            [3, 5, 0.65, 1], 1, "../")
        final_orbit_path = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.extend_orbit_by_blocks(
            3, "0.5", 50, 70, 0.1, 0.6, 4, 2,
            2, [3, 5, 0.65, 1], 1, "../")
        temp_dir = "temp_dir"
        os.rename(os.path.dirname(final_orbit_path), temp_dir)
        final_orbit_path1 = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.build_orbit_by_blocks(
            3, "0.5", 50, 70, 0.1, 0.6, 4, 2,
            [3, 5, 0.65, 1], 1, "../")
        self.assertEqual(final_orbit_path1, final_orbit_path)
        self.assertTrue(general_lib.compare_directories(os.path.dirname(final_orbit_path), temp_dir))
        # Remove the directories of the orbits created:
        shutil.rmtree(os.path.dirname(os.path.dirname(partial_orbit_path)))
        shutil.rmtree(os.path.dirname(os.path.dirname(final_orbit_path)))
        shutil.rmtree(temp_dir)
        destination_dir = "../orbits/k3A0.5/orbit_k3A0.5phi0.1p0.6prec50n_iter2/phi3to5p0.65to1"
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95F\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x01\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x08\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x01\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(G?\x9cd\xa0y\xc3\xcc\x84G?\x9cj0\xdc|\xff@NNNet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h8\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h:h<}\x94(h>h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x01\x85\x94h\x1b\x89C\x08\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhO\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      f"{destination_dir}/misc_outputs.pkl")
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\xf2\x01\x00\x00\x00\x00\x00\x00\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x03K\x03\x86\x94h\x03\x8c\x05dtype\x94\x93\x94\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(K\x00\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\rccccccccccccd\x94J\xc9\xff\xff\xffK4t\x94bh\x15)\x81\x94(K\x00\x8c\x0e13333333333333\x94J\xcb\xff\xff\xffK5t\x94bK\x01h\x15)\x81\x94(K\x00\x8c+19cb1476a9e507f08b48b0c60bcce876da5c79bab8d\x94JX\xff\xff\xffK\xa9t\x94bh\x15)\x81\x94(K\x00\x8c+180eb1dce59ff6e7470a1db4458959c369230e17ba1\x94JW\xff\xff\xffK\xa9t\x94bK\x02h\x15)\x81\x94(K\x00\x8c*98de2073be203b79f60ed7b9c04a1d35aba9344b81\x94JZ\xff\xff\xffK\xa8t\x94bh\x15)\x81\x94(K\x00\x8c*566e0ecae8b4792e0cd146de312c03883309138b85\x94JY\xff\xff\xffK\xa7t\x94bet\x94b.'),
                                      f"{destination_dir}/main_block_current.pkl")
        final_filepath = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.extend_orbit_by_blocks(
            3, "0.5", 50, 70, 0.1, 0.6, 4, 2,
            2, [3, 5, 0.65, 1], 2, "../", 3)
        actual1 = general_lib.unpickle(final_filepath)
        full_orbit1 = np.array(
            [[0, mp.mpf('0.10000000000000000555111512312578270211815834045410156'),
              mp.mpf('0.59999999999999997779553950749686919152736663818359375')],
             [1, mp.mpf('1.6120800624178759229699463278843131593259519408349936'),
              mp.mpf('0.75179379600467554117001778134832363269998336797650645')],
             [2, mp.mpf('2.3885575418758013530823756554971554447063111545136467'),
              mp.mpf('0.67523369701067162906524486253688838137369871578850557')],
             [3, mp.mpf('3.9800092720241094074161887899304061332100057787450261'),
              mp.mpf('0.63914190200217508334610692107607949045665683785813668')],
             [4, mp.mpf('4.8203530576808731353293115046360311355561032779515226'),
              mp.mpf('0.80141497901039523963131202216294403204310318208661886')]])
        np.testing.assert_array_almost_equal(actual1, full_orbit1.take([0, 4], axis=0), decimal=50)
        destination_dir1 = os.path.dirname(final_filepath)
        self.assertEqual(destination_dir1, "../orbits/k3A0.5/orbit_k3A0.5phi0.1p0.6prec50n_iter4/phi3to5p0.65to1")
        shutil.rmtree(os.path.dirname(destination_dir))
        shutil.rmtree(os.path.dirname(destination_dir1))

    def test_separatrix_by_blocks(self):
        temp_dir = "temp_dir_block_test"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if os.listdir(temp_dir):  # if the directory is not empty, clear it:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        dir_name = "separatrix_dir"
        hyperbolic_point = general_lib.unpickle("../orbits/k3A0.5/periodic_orbit_3_prec100.pkl")[0, 1:]
        # The following values have been found in study_periodic_orbits_3_contd.py:
        approach_separatrix_direction = np.array([mp.mpf(
            '0.9213025458330156785484514755760069092808056526352369060451706720081963533359324824030986355389383424'),
            mp.mpf(
                '0.5952946330866229281693117916980069475370291147621769028300174163335364031482507627320533667446453397'
            )])  # the eigenvalue is ~ 0.00151084870918, see study_periodic_orbits_3_contd.py
        other_separatrix_direction = np.array([mp.mpf(
            '0.8399202223178986835052640770685863384577150102544350359527862462712166529097310619496830577650824604'),
            mp.mpf(
                '-0.5427098857966857141525384380374198115988772355934381847888964274042643976562933146041949458764756226'
            )])  # the eigenvalue is ~ 660.346894541
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction + np.array([0, 0.01])
        starting_point2 = starting_point1 - np.array([0, 0.02])
        separatrix_path = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.separatrix_by_blocks(
            3, "0.5", 7, 12, 3, 1000, hyperbolic_point,
            approach_separatrix_direction, other_separatrix_direction, 10 ** (-3), 1, True,
            starting_point1, starting_point2, 0.1, 2, 2, 1,
            temp_dir, dir_name)
        self.assertEqual(separatrix_path, os.path.join(
            temp_dir, "orbits", "k3A0.5", "separatrices", dir_name, "separatrix.pkl"))
        filepaths = {"separatrix": f"{temp_dir}/orbits/separatrices/separatrix.pkl",
                     "temp_separatrix": f"{temp_dir}/orbits/separatrices/temp_separatrix.pkl",
                     "misc_outputs": f"{temp_dir}/orbits/separatrices/misc_outputs.pkl",
                     "temp_misc_outputs": f"{temp_dir}/orbits/separatrices/temp_misc_outputs.pkl"}
        billiard_main = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 7)
        billiard_val = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 12)
        max_discrepancy = general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, 3, 1000, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, 10 ** (-3), 1, True, starting_point1, starting_point2,
            0.1, 2, 2, 1, filepaths)
        np.testing.assert_array_equal(
            general_lib.unpickle(filepaths["separatrix"]), general_lib.unpickle(separatrix_path))
        with open(os.path.join(temp_dir, "orbits", "k3A0.5", "separatrices", dir_name,
                               pathlib.Path(__name__).with_suffix(".log")), "r") as file:
            log_contents = json.load(file)
        self.assertEqual(max_discrepancy, log_contents["max_discrepancy"])
        self.assertEqual(log_contents["params"],
                         {"k": 3, "A_exact": "0.5", "precision_main": 7, "precision_val": 12, "period": 3,
                          "iteration_limit": 1000, "hyperbolic_point": ["mpf('0.0')", "mpf('0.4698463104')"],
                          "approach_separatrix_direction": ["mpf('0.9213025458')", "mpf('0.5952946331')"],
                          "other_separatrix_direction": ["mpf('0.8399202223')", "mpf('-0.5427098858')"],
                          "precision": 0.001, "separating_coefficient": 1, "forward": True,
                          "starting_point1": ["mpf('0.09213025458')", "mpf('0.5393757737')"],
                          "starting_point2": ["mpf('0.09213025458')", "mpf('0.5193757737')"], "approaching_ratio": 0.1,
                          "num_new_orbits": 2, "block_size": 2, "num_steps_back": 1, "path_to_orbit_ambient": temp_dir,
                          "dir_name": dir_name, "verbose": False})
        actual_misc_outputs = pd.DataFrame(log_contents["progress"])
        pd.testing.assert_frame_equal(
            actual_misc_outputs.drop(columns="time_finished"),
            pd.DataFrame({"start_block": [0, 2],
                          "starting_point1": [log_contents["params"]["starting_point1"],
                                              ["mpf('0.0009213025478')", "mpf('0.4705416076')"]],
                          "starting_point2": [log_contents["params"]["starting_point2"],
                                              ["mpf('0.0009213025478')", "mpf('0.4703416079')"]],
                          "discrepancy": [max_discrepancy, np.nan]}))
        shutil.rmtree(temp_dir)

    def test_extend_validate_separatrix(self):
        temp_dir = "temp_dir_block_test"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if os.listdir(temp_dir):  # if the directory is not empty, clear it:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        dir_name = "separatrix_dir"
        hyperbolic_point = general_lib.unpickle("../orbits/k3A0.5/periodic_orbit_3_prec100.pkl")[0, 1:]
        # The following values have been found in study_periodic_orbits_3_contd.py:
        approach_separatrix_direction = np.array([mp.mpf(
            '0.9213025458330156785484514755760069092808056526352369060451706720081963533359324824030986355389383424'),
            mp.mpf(
                '0.5952946330866229281693117916980069475370291147621769028300174163335364031482507627320533667446453397'
            )])  # the eigenvalue is ~ 0.00151084870918, see study_periodic_orbits_3_contd.py
        other_separatrix_direction = np.array([mp.mpf(
            '0.8399202223178986835052640770685863384577150102544350359527862462712166529097310619496830577650824604'),
            mp.mpf(
                '-0.5427098857966857141525384380374198115988772355934381847888964274042643976562933146041949458764756226'
            )])  # the eigenvalue is ~ 660.346894541
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction + np.array([0, 0.01])
        starting_point2 = starting_point1 - np.array([0, 0.02])
        # first create separatrix with num_steps = 0, and then extend it for 1 step:
        separatrix_path = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.separatrix_by_blocks(
            3, "0.5", 7, 12, 3, 1000, hyperbolic_point,
            approach_separatrix_direction, other_separatrix_direction, 10 ** (-3), 1, True,
            starting_point1, starting_point2, 0.1, 2, 2, 0,
            temp_dir, dir_name)
        separatrix_dir = os.path.join(temp_dir, "orbits", "k3A0.5", "separatrices", dir_name)
        self.assertEqual(separatrix_path, os.path.join(separatrix_dir, "separatrix.pkl"))
        logfile = os.path.join(separatrix_dir, pathlib.Path(__name__).with_suffix(".log"))
        billiard_Birkhoff_lib.BirkhoffBilliard_k_A.extend_validate_separatrix(
            3, "0.5", 7, 12, True, temp_dir, dir_name, 1)
        actual1 = general_lib.unpickle(separatrix_path)
        with open(logfile) as file:
            message = file.read()
        # Check only the beginning of the message (before the timestamps that change from run to run)
        # and the last part of the message, created by extend_validate_separatrix(.):
        beginning_of_log = """{
    "params": {
        "k": 3,
        "A_exact": "0.5",
        "precision_main": 7,
        "precision_val": 12,
        "period": 3,
        "iteration_limit": 1000,
        "hyperbolic_point": [
            "mpf('0.0')",
            "mpf('0.4698463104')"
        ],
        "approach_separatrix_direction": [
            "mpf('0.9213025458')",
            "mpf('0.5952946331')"
        ],
        "other_separatrix_direction": [
            "mpf('0.8399202223')",
            "mpf('-0.5427098858')"
        ],
        "precision": 0.001,
        "separating_coefficient": 1,
        "forward": true,
        "starting_point1": [
            "mpf('0.09213025458')",
            "mpf('0.5393757737')"
        ],
        "starting_point2": [
            "mpf('0.09213025458')",
            "mpf('0.5193757737')"
        ],
        "approaching_ratio": 0.1,
        "num_new_orbits": 2,
        "block_size": 2,
        "num_steps_back": 0,
        "path_to_orbit_ambient": "temp_dir_block_test",
        "dir_name": "separatrix_dir",
        "verbose": false
    },
    "max_discrepancy": 1.7940328689292073e-07,
    "progress": [
        {
            "start_block": 0,
            "starting_point1": [
                "mpf('0.09213025458')",
                "mpf('0.5393757737')"
            ],
            "starting_point2": [
                "mpf('0.09213025458')",
                "mpf('0.5193757737')"
            ],
            "discrepancy":"""
        self.assertEqual(message[:len(beginning_of_log)], beginning_of_log)
        self.assertEqual(message[-108:-23], "Extended the separatrix away from the hyperbolic additional 1 steps, max_discrepancy:")
        shutil.rmtree(temp_dir)

        # compare it with doing all in one step:
        separatrix_path1 = billiard_Birkhoff_lib.BirkhoffBilliard_k_A.separatrix_by_blocks(
            3, "0.5", 7, 12, 3, 1000, hyperbolic_point,
            approach_separatrix_direction, other_separatrix_direction, 10 ** (-3), 1, True,
            starting_point1, starting_point2, 0.1, 2, 2, 1,
            temp_dir, dir_name)
        self.assertEqual(separatrix_path, separatrix_path1)
        actual2 = general_lib.unpickle(separatrix_path)
        np.testing.assert_array_equal(actual2, actual1)
        shutil.rmtree(temp_dir)


