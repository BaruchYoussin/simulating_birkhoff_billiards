import os.path
import pickle
import shutil
from unittest import TestCase

import mpmath as mp
import numpy as np
import pandas as pd

import billiard_Birkhoff_lib
import general_lib


class Test(TestCase):
    def test_parametric_from_supporting(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        point = billiard.param_boundary()(mp.mpf(0))
        self.assertAlmostEqual(mp.mpf("0.766044443118978035202392650555"), point[0], 30)
        self.assertAlmostEqual(mp.mpf(0), point[1], 30)
        self.assertNotAlmostEqual(mp.mpf("0.766044443118978035202392650555") + mp.mpf(10) ** (-29), point[0], 30)
        self.assertNotAlmostEqual(mp.mpf(10) ** (-29), point[1], 30)
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 500)
        point = billiard.param_boundary()(mp.mpf(
            "0.100000000000000005551115123125782702118158340454101562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        ))
        self.assertAlmostEqual(
            mp.mpf(
                "0.756964567999158425059567597029876433850775227178742496264990347092893031655088706261726305252054339891170380889806732631280298779772948662714632500875772594756171602665658707239083354414864250076522923915478158303731454202853825414796239524306642071928394352010933223415187636698214572512216673609734801675806160560149509251601544636073991737029275960707492084755485289784411906900108025139182088319952382159126329064244580308515170202931863958875585658560630710183148293145691483796678713695881271932"
            ), point[0], 500)
        self.assertAlmostEqual(mp.mpf(
            "0.18349298643945358055335319666967945357297032834103912270889030657791265293179432107234684196307784501202987189447906991057782066602830290278044438950779670076081186406216160593180489690417105611098726667972752990402424463292844836840245434834160707692229865573449596331692366528955026398904288443429051345123442288618887667510085391627228696853189938472446257890782600398205802602032909705313088843551138345957506794606660178387045745135493871295176150359751169263625894654131462172758954407782301455"
        ), point[1], 500)

    def test_intersect_line_with_parametric_billiard_boundary(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        parametric_boundary = billiard.param_boundary()
        self.assertAlmostEqual(mp.mpf("2.0847732720998271111401664073441415087096473471932"),
                               general_lib.intersect_line_with_parametric_billiard_boundary(
                                   parametric_boundary, p=mp.mpf(0), phi=mp.mpf(0.5)),
                               places=30)
        self.assertAlmostEqual(mp.mpf("2.0847732720998271111401664073441415087096473471932"),
                               general_lib.intersect_line_with_parametric_billiard_boundary(
                                   parametric_boundary, p=mp.mpf(0), phi=mp.mpf(0.5), forward=True),
                               places=30)
        self.assertAlmostEqual(mp.mpf("-1.11085637151326659998879396631791"),
                               general_lib.intersect_line_with_parametric_billiard_boundary(
                                   parametric_boundary, p=mp.mpf(0), phi=mp.mpf(0.5), forward=False),
                               places=30)
        self.assertAlmostEqual(mp.mpf("0.94452500154024561615375975135153501671535498779166"),
                               general_lib.intersect_line_with_parametric_billiard_boundary(
                                   parametric_boundary, p=mp.mpf(0.5), phi=mp.mpf(0)),
                               places=30)
        self.assertAlmostEqual(mp.mpf("-0.94452500154024561615375975135153501671535498779166"),
                               general_lib.intersect_line_with_parametric_billiard_boundary(
                                   parametric_boundary, p=mp.mpf(0.5), phi=mp.mpf(0), forward=False),
                               places=30)
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 200)
        parametric_boundary = billiard.param_boundary()
        self.assertAlmostEqual(mp.mpf(
            "2.08477327209982711114016640734414150870964734719317940165783924101162231361835734819277215410832906712145490871480650935560170220340552691510904873009960515708228984206766766651634450555091795053575728064661938742828937608597517068766210097419831930038716398393455710924124285827789142594664527307537213436180067308658649533"
        ),
            general_lib.intersect_line_with_parametric_billiard_boundary(
                parametric_boundary, p=mp.mpf(0), phi=mp.mpf(0.5)),
            places=200)
        self.assertAlmostEqual(mp.mpf(
            "-1.1108563715132665999887939663179061237636141253909130917485828700132750396712183933806027957677803794305410440560175060102961062992134755800174674677894347229399791274688928136591573923669354790891607957"
        ),
            general_lib.intersect_line_with_parametric_billiard_boundary(
                parametric_boundary, p=mp.mpf(0), phi=mp.mpf(0.5), forward=False),
            places=200)
        self.assertAlmostEqual(mp.mpf(
            "0.94452500154024561615375975135153501671535498779166012803466575589308041663816711957189918929212063663125160563203980257050711482121775626736631465276654581191168120349120748950849804128435353782566745789712480234655937907206403607586838786384434291851781978538242403130096308787478797220997311753993612779946645031846455053"
        ),
            general_lib.intersect_line_with_parametric_billiard_boundary(
                parametric_boundary, p=mp.mpf(0.5), phi=mp.mpf(0)),
            places=200)
        self.assertAlmostEqual(mp.mpf(
            "-0.94452500154024561615375975135153501671535498779166012803466575589308041663816711957189918929212063663125160563203980257050711482121775626736631465276654581191168120349120748950849804128435353782566745789712480234655937907206403607586838786384434291851781978538242403130096308787478797220997311753993612779946645031846455053"
        ),
            general_lib.intersect_line_with_parametric_billiard_boundary(
                parametric_boundary, p=mp.mpf(0.5), phi=mp.mpf(0), forward=False),
            places=200)

    def test_reflect_line_at_boundary(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30)
        parametric_boundary = billiard.param_boundary()
        phi, p = general_lib.reflect_line_at_boundary(mp.mpf(0), mp.mpf(0.5), parametric_boundary)
        self.assertAlmostEqual(mp.mpf("1.8890500030804912323075195027030700334307099755833"), phi, places=30)
        self.assertAlmostEqual(mp.mpf("0.5979116727228235764163996027748549242762824519879"), p, places=30)
        phi, p = general_lib.reflect_line_at_boundary(mp.mpf(0), mp.mpf(0.5), parametric_boundary, forward=False)
        self.assertAlmostEqual(2 * mp.pi - mp.mpf("1.8890500030804912323075195027030700334307099755833"), phi, places=30)
        self.assertAlmostEqual(mp.mpf("0.5979116727228235764163996027748549242762824519879"), p, places=30)
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 200)
        parametric_boundary = billiard.param_boundary()
        phi, p = general_lib.reflect_line_at_boundary(mp.mpf(0), mp.mpf(0.5), parametric_boundary)
        self.assertAlmostEqual(mp.mpf(
            "1.88905000308049123230751950270307003343070997558332025606933151178616083327633423914379837858424127326250321126407960514101422964243551253473262930553309162382336240698241497901699608256870707565133491579424960469311875814412807215173677572768868583703563957076484806260192617574957594441994623507987225559893290063692910"
        ), phi, places=30)
        self.assertAlmostEqual(mp.mpf(
            "0.59791167272282357641639960277485492427628245198793402105724323781121350963063024367868750425826788137583762154062993510911148622137827995546640340658884766881734653629112870069569392267396424997920946619907616812873912581697924575703469949560666136007240330738318114618570387671183867974697270222719528703387739052825127"
        ), p, places=30)
        # test whether max_iter is enough (if not, findroot throws an exception):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 30000)
        parametric_boundary = billiard.param_boundary()
        phi, p = general_lib.reflect_line_at_boundary(mp.mpf(0), mp.mpf(0.5), parametric_boundary)

    def test_iterate_function(self):
        def fn(x):
            return x + 1

        self.assertSequenceEqual(list(range(11)), general_lib.iterate_function(fn, 0, 10))

        def fn(x, a):
            return x + a

        self.assertSequenceEqual(list(range(0, 22, 2)), general_lib.iterate_function(fn, 0, 10, 2))

        def fn(x, a, b):
            return x + a + b

        self.assertSequenceEqual(list(range(0, 22, 2)), general_lib.iterate_function(fn, 0, 10, 1, 1))
        self.assertSequenceEqual(list(range(0, 22, 2)), general_lib.iterate_function(fn, 0, 10, 1, b = 1))
        self.assertSequenceEqual(list(range(0, 22, 2)), general_lib.iterate_function(fn, 0, 10, a = 1, b = 1))

        def fn(x, y, a, b):
            return x + a, y + b

        self.assertSequenceEqual(list(zip(range(11), range(1, 12))), general_lib.iterate_function(fn, (0, 1), 10, 1, 1))
        self.assertSequenceEqual(list(zip(range(11), range(1, 12))), general_lib.iterate_function(fn, (0, 1), 10,
                                                                                                  b = 1, a = 1))

    def test_iterate_function_until_condition(self):
        def fn(x):
            return x + 1

        self.assertEqual((None, list(range(11))), general_lib.iterate_function_until_condition(
            fn, 0, lambda x: None, 10))
        self.assertEqual((1, list(range(7))),
                         general_lib.iterate_function_until_condition(
                             fn, 0, lambda x: 1 if x > 5 else None, 10))
        self.assertEqual((1, list(range(7))),
                         general_lib.iterate_function_until_condition(
                             fn, 0, lambda x: -1 if x > 7 else 1 if x > 5 else None, 10))
        self.assertEqual((None, list(range(11))),
                         general_lib.iterate_function_until_condition(
                             fn, 0, lambda x: 1 if x > 10 else None, 10))


    def test_calculate_orbit(self):
        billiard50 = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        parametric_boundary = billiard50.param_boundary()
        init_phi = 0.1
        init_p = 0.6
        phi1, p1 = general_lib.reflect_line_at_boundary(init_phi, init_p, parametric_boundary)
        phi2, p2 = general_lib.reflect_line_at_boundary(phi1, p1, parametric_boundary)
        phi3, p3 = general_lib.reflect_line_at_boundary(phi2, p2, parametric_boundary)
        orbit1 = general_lib.calculate_orbit(billiard50, init_phi, init_p, 3)
        print(orbit1)  # mp.nstr(..) does not help: it does not go into np.array.
        # [[0 mpf('0.10000000000000000555111512312578270211815834045410156')
        #   mpf('0.59999999999999997779553950749686919152736663818359375')]
        #  [1 mpf('1.6120800624178759229699463278843131593259519408349936')
        #   mpf('0.75179379600467554117001778134832363269998336797650645')]
        #  [2 mpf('2.3885575418758013530823756554971554447063111545136467')
        #   mpf('0.67523369701067162906524486253688838137369871578850691')]
        #  [3 mpf('3.9800092720241094074161887899304061332100057787450154')
        #   mpf('0.6391419020021750833461069210760794904566568378581447')]]
        expected1 = np.array([(0, init_phi, init_p), (1, phi1, p1), (2, phi2, p2), (3, phi3, p3)])
        self.assertTrue(np.array_equal(expected1, orbit1))
        orbit2 = general_lib.calculate_orbit(billiard50, init_phi, init_p, 3, start_numbering=10)
        print(orbit2)
        # [[10 mpf('0.10000000000000000555111512312578270211815834045410156')
        #   mpf('0.59999999999999997779553950749686919152736663818359375')]
        #  [11 mpf('1.6120800624178759229699463278843131593259519408349936')
        #   mpf('0.75179379600467554117001778134832363269998336797650645')]
        #  [12 mpf('2.3885575418758013530823756554971554447063111545136467')
        #   mpf('0.67523369701067162906524486253688838137369871578850557')]
        #  [13 mpf('3.9800092720241094074161887899304061332100057787450261')
        #   mpf('0.63914190200217508334610692107607949045665683785813668')]]
        self.assertTrue(np.array_equal(
            np.array([(10, init_phi, init_p), (11, phi1, p1), (12, phi2, p2), (13, phi3, p3)]), orbit2))
        # calculate the same orbit backwards, starting from the last point:
        orbit_backwards = general_lib.calculate_orbit(billiard50, phi3, p3, 3, forward=False)
        # losing one decimal in precision, 49 instead of 50:
        np.testing.assert_array_almost_equal(
            np.array([(0, phi3, p3), (1, phi2, p2), (2, phi1, p1), (3, init_phi, init_p)]), orbit_backwards, decimal=49)

        billiard100 = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 100)
        orbit3 = general_lib.calculate_orbit(billiard100, init_phi, init_p, 3)
        print(orbit3)
        expected3 = np.array(
            [[0, mp.mpf('0.1000000000000000055511151231257827021181583404541015625'),
              mp.mpf('0.59999999999999997779553950749686919152736663818359375')],
             [1,
              mp.mpf(
                  '1.612080062417875922969946327884313159325951940834993883981063959348987720394692881563769722649170417091'),
              mp.mpf(
                  '0.7517937960046755411700177813483236326999833679765079890837823332334720885555339279076911506365352680107')],
             [2,
              mp.mpf(
                  '2.388557541875801353082375655497155444706311154513641419212045322617295866367409249079609121725922688194'),
              mp.mpf(
                  '0.6752336970106716290652448625368883813736987157885076861174457949826798254950306492494194071700572440043')],
             [3,
              mp.mpf(
                  '3.980009272024109407416188789930406133210005778744989011621410609177483280983819511212245334544389827178'),
              mp.mpf(
                  '0.6391419020021750833461069210760794904566568378581589900387874368180411801971063852192973219289369334257')]]
        )
        self.assertTrue(np.array_equal(expected3, orbit3))
        # Testing that alternating between using billiard50 and billiard100 yields consistent results:
        orbit4 = general_lib.calculate_orbit(billiard50, init_phi, init_p, 3)
        print(orbit4)
        self.assertTrue(np.array_equal(orbit4, orbit1))
        orbit5 = general_lib.calculate_orbit(billiard100, init_phi, init_p, 3)
        expected5 = np.array(
            [[0, mp.mpf('0.1000000000000000055511151231257827021181583404541015625'),
              mp.mpf('0.59999999999999997779553950749686919152736663818359375')],
             [1,
              mp.mpf(
                  '1.612080062417875922969946327884313159325951940834993883981063959348987720394692881563769722649170417091'),
              mp.mpf(
                  '0.7517937960046755411700177813483236326999833679765079890837823332334720885555339279076911506365352680107')],
             [2,
              mp.mpf(
                  '2.388557541875801353082375655497155444706311154513641419212045322617295866367409249079609121725922688194'),
              mp.mpf(
                  '0.6752336970106716290652448625368883813736987157885076861174457949826798254950306492494194071700572440043')],
             [3,
              mp.mpf(
                  '3.980009272024109407416188789930406133210005778744989011621410609177483280983819511212245334544389827178'),
              mp.mpf(
                  '0.6391419020021750833461069210760794904566568378581589900387874368180411801971063852192973219289369334257')]]
        )
        self.assertTrue(np.array_equal(expected5, orbit5))

    def test_linspace(self):
        mp.mp.dps = 15
        self.assertEqual(general_lib.linspace(1, 10, 10), mp.linspace(1, 10, 10))
        self.assertEqual([str(x) for x in general_lib.linspace(-0.2, 1, 7)],
                         ["-0.2", "0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        mp.mp.dps = 50
        self.assertEqual([str(x) for x in general_lib.linspace("-0.2", "1", 7)],
                         ["-0.2", "0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    def test_two_sided_range(self):
        mp.mp.dps = 50
        self.assertEqual([str(x) for x in general_lib.two_sided_range("-0.2", "1", "0.2")],
                         ["-0.2", "0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    def test_get_residues_to_left_bound(self):
        mp.mp.dps = 15
        np.testing.assert_array_equal(
            general_lib.get_residues_to_left_bound(np.array([mp.mpf(1), mp.mpf(2)]), 1.5, 2),
            np.array([mp.mpf(3), mp.mpf(2)]))
        np.testing.assert_array_equal(
            general_lib.get_residues_to_left_bound(np.array([mp.mpf(1), mp.mpf(2)]), mp.mpf(0.5), mp.mpf(2)),
            np.array([mp.mpf(1), mp.mpf(2)]))
        np.testing.assert_array_equal(
            general_lib.get_residues_to_left_bound(np.array([mp.mpf(1), mp.mpf(2)]), mp.mpf(-1.5), 2),
            np.array([mp.mpf(-1), mp.mpf(0)]))
        np.testing.assert_array_equal(
            general_lib.get_residues_to_left_bound(np.array([mp.mpf(1), mp.mpf(2)]), -3, mp.mpf(2)),
            np.array([mp.mpf(-3), mp.mpf(-2)]))
        np.testing.assert_array_equal(
            general_lib.get_residues_to_left_bound(np.array([mp.mpf(1), mp.mpf(2)]), 0, mp.mpf(2)),
            np.array([mp.mpf(1), mp.mpf(0)]))


    def test_clip_pair_of_arrays(self):
        mp.mp.dps = 15
        x = np.array([0.1, 2.3, 0.3, 2.5])
        y = np.array([0.2, 3.4, 3.1, 0.5])
        xclipped, yclipped = general_lib.clip_pair_of_arrays(x, y, 0, 1, 0, 4)
        self.assertTrue(np.array_equal(np.array([0.1, 0.3]), xclipped))
        self.assertTrue(np.array_equal(np.array([0.2, 3.1]), yclipped))
        xclipped, yclipped = general_lib.clip_pair_of_arrays(x, y, 0, 4, 0, 1)
        self.assertTrue(np.array_equal(np.array([0.1, 2.5]), xclipped))
        self.assertTrue(np.array_equal(np.array([0.2, 0.5]), yclipped))
        mp.mp.dps = 50
        x = np.array([mp.mpf(0.1), mp.mpf(2.3), mp.mpf(0.3), mp.mpf(2.5)])
        y = np.array([mp.mpf(0.2), mp.mpf(3.4), mp.mpf(3.1), mp.mpf(0.5)])
        xclipped, yclipped = general_lib.clip_pair_of_arrays(x, y, mp.mpf(0), mp.mpf(1), mp.mpf(0), mp.mpf(4))
        self.assertTrue(np.array_equal(np.array([mp.mpf(0.1), mp.mpf(0.3)]), xclipped))
        self.assertTrue(np.array_equal(np.array([mp.mpf(0.2), mp.mpf(3.1)]), yclipped))
        xclipped, yclipped = general_lib.clip_pair_of_arrays(x, y, mp.mpf(0), mp.mpf(4), mp.mpf(0), mp.mpf(1))
        self.assertTrue(np.array_equal(np.array([mp.mpf(0.1), mp.mpf(2.5)]), xclipped))
        self.assertTrue(np.array_equal(np.array([mp.mpf(0.2), mp.mpf(0.5)]), yclipped))
        xclipped, yclipped = general_lib.clip_pair_of_arrays(x, y, mp.mpf(-0.5), mp.mpf(0.5), mp.mpf(0), mp.mpf(1),
                                                             3)
        self.assertTrue(np.array_equal(np.array([mp.mpf(0.1), mp.mpf(-0.5)]), xclipped))
        self.assertTrue(np.array_equal(np.array([mp.mpf(0.2), mp.mpf(0.5)]), yclipped))

    def test_clip_orbit(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        init_phi = 0.1
        init_p = 0.6
        orbit = general_lib.calculate_orbit(billiard, init_phi, init_p, 3)
        # see this orbit printed in test_calculate_orbit
        clipped = general_lib.clip_orbit(orbit, 0, 3, 0, 1)
        self.assertTrue(np.array_equal(orbit[:3, :], clipped))
        clipped = general_lib.clip_orbit(orbit, 0, 4, 0.61, 1)
        self.assertTrue(np.array_equal(orbit[1:, :], clipped))
        clipped = general_lib.clip_orbit(orbit, 0.2, 4, 0, 0.7)
        self.assertTrue(np.array_equal(orbit[2:, :], clipped))
        clipped = general_lib.clip_orbit(orbit, 1, 2, 0, 0.7)
        self.assertTrue(np.array_equal(orbit[:0, :], clipped))
        clipped = general_lib.clip_orbit(orbit, 0, 4, 0, 1)
        self.assertTrue(np.array_equal(orbit, clipped))

    def test_extend_orbit(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        parametric_boundary = billiard.param_boundary()
        init_phi = 0.1
        init_p = 0.6
        orbit = general_lib.calculate_orbit(billiard, init_phi, init_p, 3)
        # see this orbit printed in test_calculate_orbit
        self.assertTrue(np.array_equal(orbit, general_lib.extend_orbit(orbit, None)))
        extended = general_lib.extend_orbit(orbit, {
            "previous_points": np.array([[0, 1, 2], [3, 4, 5]]), "rectangle": [0, 3, 0, 1]})
        expected = np.array([[0, 1, 2], [3, 4, 5],
                             [1, mp.mpf('1.6120800624178759229699463278843131593259519408349936'),
                              mp.mpf('0.75179379600467554117001778134832363269998336797650645')],
                             [2, mp.mpf('2.3885575418758013530823756554971554447063111545136467'),
                              mp.mpf('0.67523369701067162906524486253688838137369871578850557')]])
        np.testing.assert_array_almost_equal(expected, extended, decimal=50)
        extended = general_lib.extend_orbit(orbit, {
            "previous_points": None, "rectangle": [0, 3, 0, 1]})
        self.assertTrue(np.array_equal(orbit[:3, :], extended))
        extended = general_lib.extend_orbit(orbit, {
            "previous_points": None, "rectangle": ["0", "3", "0", "1"]})
        self.assertTrue(np.array_equal(orbit[:3, :], extended))
        extended = general_lib.extend_orbit(orbit, {
            "previous_points": None,
            "rectangle": ["1.61208006241787592296994632788431315932595",
                          "1.61208006241787592296994632788431315932596",
                          "0.751793796004675541170017781348323632699983367",
                          "0.751793796004675541170017781348323632699983368"]})
        np.testing.assert_array_almost_equal(orbit[:2, :], extended, decimal=49)
        extended = general_lib.extend_orbit(orbit, {
            "previous_points": None,
            "rectangle": ["1.612080062417875922969946327884313159325952",
                          "1.61208006241787592296994632788431315932596",
                          "0.751793796004675541170017781348323632699983367",
                          "0.751793796004675541170017781348323632699983368"]})
        self.assertTrue(np.array_equal(orbit[:1, :], extended))

    def test_rescale_array(self):
        array = np.array([0.1, 2.3, 0.3, 2.5])
        self.assertTrue(np.allclose(np.array([-0.9, 1.3, -0.7, 1.5]), general_lib.rescale_array(array, 1, 2),
                                    rtol=0, atol=1e-15))
        array = np.array([0.2, 3.4, 3.1, 0.5])
        self.assertTrue(np.allclose(np.array([0.05, 0.85, 0.775, 0.125]), general_lib.rescale_array(array, 0, 4),
                                    rtol=0, atol=1e-15))
        mp.mp.dps = 50
        array = np.array([mp.mpf(0.1), mp.mpf(2.3), mp.mpf(0.3), mp.mpf(2.5)])
        self.assertTrue(np.abs(np.array([mp.mpf(-0.9), mp.mpf(1.3), mp.mpf(-0.7), mp.mpf(1.5)])
                                    - general_lib.rescale_array(array, mp.mpf(1), mp.mpf(2))).max() < 1e-15)
        array = np.array([mp.mpf(0.2), mp.mpf(3.4), mp.mpf(3.1), mp.mpf(0.5)])
        self.assertTrue(np.abs(np.array([mp.mpf(0.05), mp.mpf(0.85), mp.mpf(0.775), mp.mpf(0.125)])
                               - general_lib.rescale_array(array, mp.mpf(0), mp.mpf(4))).max() < 1e-15)

    def test_rescale_array_to_float(self):
        mp.mp.dps = 50
        array = np.array([mp.mpf(0.1), mp.mpf(2.3), mp.mpf(0.3), mp.mpf(2.5)])
        rescaled_array = general_lib.rescale_array_to_float(array, mp.mpf(1), mp.mpf(2))
        np.testing.assert_almost_equal(rescaled_array, np.array([-0.9, 1.3, -0.7, 1.5]), decimal=14)
        self.assertTrue(rescaled_array.dtype == float)
        array = np.array([mp.mpf(0.2), mp.mpf(3.4), mp.mpf(3.1), mp.mpf(0.5)])
        rescaled_array = general_lib.rescale_array_to_float(array, mp.mpf(0), mp.mpf(4))
        np.testing.assert_almost_equal(rescaled_array, np.array([0.05, 0.85, 0.775, 0.125]), decimal=14)
        self.assertTrue(rescaled_array.dtype == float)

    def test_circled_discrepancy(self):
        array_1 = np.array([1, 2, 3, 5])
        array_2 = np.array([0.5, 2.3, 0, 2])
        self.assertEqual(0.5, general_lib.circled_discrepancy(array_1, array_2, 3.1))
        array_2 = np.array([1, 2, 0, 2])
        self.assertAlmostEqual(0.1, general_lib.circled_discrepancy(array_1, array_2, 3.1), places=10)
        mp.mp.dps = 50
        array_1 = np.array([1, 2, 3, 2 * mp.pi - 0.1])
        array_2 = np.array([1, 2, 3, 0])
        self.assertAlmostEqual(0.1, general_lib.circled_discrepancy(array_1, array_2, 2 * mp.pi))

    def test_circled_signed_discrepancy(self):
        self.assertEqual(general_lib.circled_signed_discrepancy(1, 3, None), 2)
        self.assertEqual(general_lib.circled_signed_discrepancy(1, 3, 2), 0)
        self.assertEqual(general_lib.circled_signed_discrepancy(1, 3, 2.5), -0.5)
        self.assertEqual(general_lib.circled_signed_discrepancy(2, 0, 2.5), 0.5)
        self.assertEqual(general_lib.circled_signed_discrepancy(mp.mpf(1), mp.mpf(0), 2.5), -1)

    def test_create_validate_orbit_by_blocks(self):
        temp_dir = "temp_dir_block_test"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if os.listdir(temp_dir):  # if the directory is not empty, clear it:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        filepaths = {"clipped_orbit_previous": f"{temp_dir}/clipped_orbit_previous.pkl",
                     "clipped_orbit_current": f"{temp_dir}/orbit.pkl",
                     "main_block_previous": f"{temp_dir}/main_block_previous.pkl",
                     "validation_block_previous": f"{temp_dir}/validation_block_previous.pkl",
                     "main_block_current": f"{temp_dir}/main_block_current.pkl",
                     "validation_block_current": f"{temp_dir}/validation_block_current.pkl",
                     "misc_outputs": f"{temp_dir}/misc_outputs.pkl"}
        billiard_main = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        billiard_val = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 70)
        # Check that the result is correct in case of one block:
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 4,
                                                    billiard_main, billiard_val, [0, 4, 0.61, 1], filepaths)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        full_orbit_main = general_lib.calculate_orbit(billiard_main, 0.1, 0.6, 4)
        print(full_orbit_main)  # output:
        # [[0 mpf('0.10000000000000000555111512312578270211815834045410156')
        #   mpf('0.59999999999999997779553950749686919152736663818359375')]
        #  [1 mpf('1.6120800624178759229699463278843131593259519408349936')
        #   mpf('0.75179379600467554117001778134832363269998336797650645')]
        #  [2 mpf('2.3885575418758013530823756554971554447063111545136467')
        #   mpf('0.67523369701067162906524486253688838137369871578850691')]
        #  [3 mpf('3.9800092720241094074161887899304061332100057787450154')
        #   mpf('0.6391419020021750833461069210760794904566568378581447')]
        #  [4 mpf('4.8203530576808731353293115046360311355561032779514905')
        #   mpf('0.8014149790103952396313120221629440320431031820866095')]]
        expected1 = full_orbit_main[:4, :]
        self.assertTrue(np.array_equal(expected1, actual1))
        shutil.rmtree(temp_dir)
        # Check that the result is correct in case of two blocks:
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [0, 4, 0.61, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        self.assertTrue(np.array_equal(expected1, actual1))
        # Check that validation picks up correctly in the middle, with no precision loss due to the break:
        full_orbit_val = general_lib.calculate_orbit(billiard_val, 0.1, 0.6, 4)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        actual_table1 = general_lib.unpickle(filepaths["misc_outputs"])
        # check the index and the block_end column:
        pd.testing.assert_frame_equal(pd.DataFrame(
            {"block_end": [2, 4]}, index=pd.Index([0, 2], name="block_start"), dtype="float"),
            actual_table1[["block_end"]])
        # check the discrepancy column:
        [discrepancy1, discrepancy2] = actual_table1["discrepancy"].to_list()
        self.assertTrue(1e-55 < discrepancy1 < 1e-49)
        self.assertTrue(1e-55 < discrepancy2 < 1e-45)
        os.remove(filepaths["misc_outputs"])
        # Change the rectangle:
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        expected1 = full_orbit_main.take([0, 4], axis=0)
        np.testing.assert_array_equal(expected1, actual1)
        shutil.rmtree(temp_dir)
        # In the same example, start in the first pass of the loop with starting_step = 2
        # (starting_step = 1 on the first pass is impossible):
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95F\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x01\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x08\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x01\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(G?\x9cd\xa0y\xc3\xcc\x84G?\x9cj0\xdc|\xff@NNNet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h8\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h:h<}\x94(h>h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x01\x85\x94h\x1b\x89C\x08\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhO\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_current"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 3 on the first pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95F\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x01\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x08\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x01\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(G?\x9cd\xa0y\xc3\xcc\x84G?\x9cj0\xdc|\xff@NNNet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h8\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h:h<}\x94(h>h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x01\x85\x94h\x1b\x89C\x08\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhO\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:1, :], filepaths["clipped_orbit_current"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 4 on the first pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95V\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x01\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x08\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x01\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(G?\x9cS\xeb(qR\x17G?\x9c\\\x94\xb4\n;\xccG?\x9c5T\xd4 [\xc9G?\x9c;\xe3[\x03\xb1\xbbNet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h8\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h:h<}\x94(h>h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x01\x85\x94h\x1b\x89C\x08\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhO\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:1, :], filepaths["clipped_orbit_current"])
        general_lib.pickle_one_object(full_orbit_val[:3, :], filepaths["validation_block_current"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 5 on the first pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\x88\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x01\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x08\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x01\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(G?\x9b\x8a\x8b\xcen\xa7AG?\x9b\x8azl\xd8\x9bbG?\x9aj\x91\xd5f4\xa4G?\x9ah\x9f\xc0i\x00\x0e\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\x011\x94JY\xff\xff\xffK\x01t\x94bet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h>\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h@hB}\x94(hDh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x01\x85\x94h\x1b\x89C\x08\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhU\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:1, :], filepaths["clipped_orbit_current"])
        general_lib.pickle_one_object(full_orbit_val[:3, :], filepaths["validation_block_current"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 1 on the second pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\x88\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x01\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x08\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x01\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(G?\x9b\x8a\x8b\xcen\xa7AG?\x9b\x8azl\xd8\x9bbG?\x9aj\x91\xd5f4\xa4G?\x9ah\x9f\xc0i\x00\x0e\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\x011\x94JY\xff\xff\xffK\x01t\x94bet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h>\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h@hB}\x94(hDh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x01\x85\x94h\x1b\x89C\x08\x00\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhU\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:1, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 2 on the second pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\xe7\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x02\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x10@\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x02\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x88]\x94(G?\x9d\xf6\xe5`R\xb7CG?\xa3\xc7f[~\xab\xd8G?\x9d\xfd\x0e*\x87\x8apG?\xa3\xc7\x94\xc3_\xd5lG?\x9c\xb2\xe4:\xc9\xf7\xd3G\x7f\xf8\x00\x00\x00\x00\x00\x00G?\x9c\xb2\xfaN\xf7E-G\x7f\xf8\x00\x00\x00\x00\x00\x00\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\x011\x94JY\xff\xff\xffK\x01t\x94bG\x7f\xf8\x00\x00\x00\x00\x00\x00et\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h>\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h@hB}\x94(hDh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x02\x85\x94h\x18\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03h\x1cNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhU\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[2:, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:1, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 3 on the second pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\xe7\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x02\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x10@\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x02\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x88]\x94(G?\x9d\xf6\xe5`R\xb7CG?\xa3\xc7f[~\xab\xd8G?\x9d\xfd\x0e*\x87\x8apG?\xa3\xc7\x94\xc3_\xd5lG?\x9c\xb2\xe4:\xc9\xf7\xd3G\x7f\xf8\x00\x00\x00\x00\x00\x00G?\x9c\xb2\xfaN\xf7E-G\x7f\xf8\x00\x00\x00\x00\x00\x00\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\x011\x94JY\xff\xff\xffK\x01t\x94bG\x7f\xf8\x00\x00\x00\x00\x00\x00et\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h>\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h@hB}\x94(hDh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x02\x85\x94h\x18\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03h\x1cNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhU\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[2:, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[[0, 4], :], filepaths["clipped_orbit_current"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:1, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 4 on the second pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\xe7\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x02\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x10@\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x02\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x88]\x94(G?\xa4e\xdd\xf7\xb9\xd7\xcaG?\xa5S\xe2]\xe1:KG?\xa4j\xb0\x80\x99v\x11G?\xa5S\xf3\xeajc\xeeG?\x9bhi\xc34\x94\xffG?\xae\xde\xe1C\xc9\xa5\xc8G?\x9bh\xee^\xa0|\xebG?\xae\xdc[\x1bj\xd0\xc0\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\x011\x94JY\xff\xff\xffK\x01t\x94bG\x7f\xf8\x00\x00\x00\x00\x00\x00et\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94h>\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94h@hB}\x94(hDh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x02\x85\x94h\x18\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03h\x1cNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhU\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[2:, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[[0, 4], :], filepaths["clipped_orbit_current"])
        general_lib.pickle_one_object(full_orbit_val[2:, :], filepaths["validation_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:1, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                    billiard_main, billiard_val, [3, 5, 0.65, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1)
        np.testing.assert_array_equal(full_orbit_val[2:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # same with starting_step = 5 on the second pass:
        general_lib.pickle_one_object(pickle.loads(b'\x80\x04\x95\xf4\x03\x00\x00\x00\x00\x00\x00\x8c\x11pandas.core.frame\x94\x8c\tDataFrame\x94\x93\x94)\x81\x94}\x94(\x8c\x04_mgr\x94\x8c\x1epandas.core.internals.managers\x94\x8c\x0cBlockManager\x94\x93\x94\x8c\x16pandas._libs.internals\x94\x8c\x0f_unpickle_block\x94\x93\x94\x8c\x16numpy._core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x01K\x02\x86\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x10@\x94t\x94b\x8c\x08builtins\x94\x8c\x05slice\x94\x93\x94K\x00K\x01K\x01\x87\x94R\x94K\x02\x87\x94R\x94h\x0bh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x05K\x02\x86\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x88]\x94(G?\xa1\x1b\xffS\xcct\xa7G?\xa5\x9f\xb6q\xca%[G?\xa1\x1c\xe7\xb9\xaf\x86\x0cG?\xa5\x93\xff\xebv\xf5\xc5G?\x9af\x85\x11\x99\xa7\xd4G?\xac\xc7\x98\x15\xc7"\xd0G?\x9aeK\xbc\xca~\xc7G?\xac\xce$\xa1\xdc\x19\x82\x8c\x14mpmath.ctx_mp_python\x94\x8c\x03mpf\x94\x93\x94)\x81\x94(K\x00\x8c\x011\x94JY\xff\xff\xffK\x01t\x94bh3)\x81\x94(K\x00\x8c\x019\x94JZ\xff\xff\xffK\x04t\x94bet\x94bh"K\x01K\x06K\x01\x87\x94R\x94K\x02\x87\x94R\x94\x86\x94]\x94(\x8c\x18pandas.core.indexes.base\x94\x8c\n_new_Index\x94\x93\x94hA\x8c\x05Index\x94\x93\x94}\x94(\x8c\x04data\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x06\x85\x94h\x18\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03h.NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\tblock_end\x94\x8c\x0eperf_time_main\x94\x8c\x11process_time_main\x94\x8c\rperf_time_val\x94\x8c\x10process_time_val\x94\x8c\x0bdiscrepancy\x94et\x94b\x8c\x04name\x94Nu\x86\x94R\x94hChE}\x94(hGh\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x02\x85\x94h\x18\x8c\x02i8\x94\x89\x88\x87\x94R\x94(K\x03h\x1cNNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89C\x10\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x94t\x94bhX\x8c\x0bblock_start\x94u\x86\x94R\x94e\x86\x94R\x94\x8c\x04_typ\x94\x8c\tdataframe\x94\x8c\t_metadata\x94]\x94\x8c\x05attrs\x94}\x94\x8c\x06_flags\x94}\x94\x8c\x17allows_duplicate_labels\x94\x88sub.'),
                                      filepaths["misc_outputs"])
        error_message = None
        try:
            general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 2,
                                                        billiard_main, billiard_val, [3, 5, 0.65, 1],
                                                        filepaths)
        except RuntimeError as e:
            error_message = str(e)
        self.assertEqual(error_message, "The table is complete, we are apparently done.")
        shutil.rmtree(temp_dir)
        # test the 3rd pass (as the tests in the first two passes missed a bug). Use the same data with block_size = 1:
        # starting_step = 1:
        misc_table = pd.DataFrame(
            {"block_end": [1., 2], "perf_time_main": [4.27e-6, 5.89e-6], "process_time_main": [4.31e-6, 5.89e-6],
             "perf_time_val": [2.62e-6, 5.91e-6], "process_time_val": [2.62e-6, 5.91e-6],
             "discrepancy": [mp.mpf(1.53e-51), mp.mpf(5.34e-51)], "block_start": [0, 1]}).set_index(
            "block_start").astype(
            {"block_end": float, "perf_time_main": "O", "process_time_main": "O",   "perf_time_val": "O",
             "process_time_val": "O", "discrepancy": "O"})
        general_lib.pickle_one_object(misc_table, filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[1:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[1:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 1,
                                                    billiard_main, billiard_val, [0, 4, 0, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        expected1 = full_orbit_main[:4, :]
        np.testing.assert_array_equal(expected1, actual1,)
        np.testing.assert_array_equal(full_orbit_val[3:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # starting_step = 2:
        misc_table.loc[2] = pd.Series([3., 6e-6, 7e-6], index=["block_end", "perf_time_main", "process_time_main"])
        general_lib.pickle_one_object(misc_table, filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[2:4, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[1:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[1:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 1,
                                                    billiard_main, billiard_val, [0, 4, 0, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1,)
        np.testing.assert_array_equal(full_orbit_val[3:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # starting_step = 3:
        general_lib.pickle_one_object(misc_table, filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[2:4, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:4, :], filepaths["clipped_orbit_current"])
        general_lib.pickle_one_object(full_orbit_main[1:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[1:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 1,
                                                    billiard_main, billiard_val, [0, 4, 0, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1,)
        np.testing.assert_array_equal(full_orbit_val[3:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # starting_step = 4:
        misc_table.at[2, "perf_time_val"] = 6.5e-6
        misc_table.at[2, "process_time_val"] = 6.5e-6
        general_lib.pickle_one_object(misc_table, filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[2:4, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:4, :], filepaths["clipped_orbit_current"])
        general_lib.pickle_one_object(full_orbit_val[2:4, :], filepaths["validation_block_current"])
        general_lib.pickle_one_object(full_orbit_main[1:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[1:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 1,
                                                    billiard_main, billiard_val, [0, 4, 0, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1,)
        np.testing.assert_array_equal(full_orbit_val[3:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)
        # starting_step = 5:
        misc_table.at[2, "discrepancy"] = mp.mpf(1.4e-50)
        general_lib.pickle_one_object(misc_table, filepaths["misc_outputs"])
        general_lib.pickle_one_object(full_orbit_main[2:4, :], filepaths["main_block_current"])
        general_lib.pickle_one_object(full_orbit_main[:4, :], filepaths["clipped_orbit_current"])
        general_lib.pickle_one_object(full_orbit_val[2:4, :], filepaths["validation_block_current"])
        general_lib.pickle_one_object(full_orbit_main[1:3, :], filepaths["main_block_previous"])
        general_lib.pickle_one_object(full_orbit_main[:3, :], filepaths["clipped_orbit_previous"])
        general_lib.pickle_one_object(full_orbit_val[1:3, :], filepaths["validation_block_previous"])
        general_lib.create_validate_orbit_by_blocks(0.1, 0.6, 4, 1,
                                                    billiard_main, billiard_val, [0, 4, 0, 1], filepaths,
                                                    keep_temp_files=True)
        actual1 = general_lib.unpickle(filepaths["clipped_orbit_current"])
        np.testing.assert_array_equal(expected1, actual1,)
        np.testing.assert_array_equal(full_orbit_val[3:, :],
                                      general_lib.unpickle(filepaths["validation_block_current"]))
        shutil.rmtree(temp_dir)

    def test_count_num_of_full_circles(self):
        self.assertEqual(general_lib.count_num_of_full_circles([1]), 0)
        self.assertEqual(general_lib.count_num_of_full_circles([1, 2]), 0)
        self.assertEqual(general_lib.count_num_of_full_circles([2, 1]), 0)
        self.assertEqual(general_lib.count_num_of_full_circles([1, 2, 1]), 1)
        self.assertEqual(general_lib.count_num_of_full_circles(np.array([1, 2, 1.1])), 1)
        self.assertEqual(general_lib.count_num_of_full_circles([1, 2, 0]), 0)
        self.assertEqual(general_lib.count_num_of_full_circles([1, 2, 0, 2]), 1)
        self.assertEqual(general_lib.count_num_of_full_circles([1, 2, 0, 2, 3]), 1)
        self.assertEqual(general_lib.count_num_of_full_circles([1, 2, 0, 2, 0]), 1)
        self.assertEqual(general_lib.count_num_of_full_circles([1, 2, 0, 2, 0, 2]), 2)
        self.assertEqual(general_lib.count_num_of_full_circles([2, 1, 2, 0, 2, 0, 2]), 3)

    def test_circular_length(self):
        self.assertEqual(general_lib.circular_length([1], 3), 0)
        self.assertEqual(general_lib.circular_length([1, 2], 3.5), 1)
        self.assertEqual(general_lib.circular_length([2, 1], 2.5), 1.5)
        self.assertEqual(general_lib.circular_length([1, 2, 1], 2), 2)
        self.assertEqual(general_lib.circular_length(np.array([1, 2, 1.1]), 3), 3.1)
        self.assertEqual(general_lib.circular_length([1, 2, 0], 2), 1)
        self.assertEqual(general_lib.circular_length([1, 2, 0, 2], 3.5), 4.5)
        self.assertEqual(general_lib.circular_length([1, 2, 0, 2, 3], 10), 12)
        self.assertEqual(general_lib.circular_length([1, 2, 0, 2, 0], 2), 1)  # 2 is replaced by 0
        self.assertEqual(general_lib.circular_length([1, 2, 0, 2, 0, 2], 3), 7)
        self.assertEqual(general_lib.circular_length([2, 1, 2, 0, 2, 0, 2], 4), 12)

    def test_periodic_orbit_error(self):
        billiard50 = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        err_phi, err_p = general_lib.periodic_orbit_error(billiard50, 3, 0.1, 0.6, 2 * mp.pi)
        self.assertEqual(err_phi, mp.mpf('3.8800092720241094018650736668046234310918474382909139') - 2 * mp.pi)
        self.assertEqual(err_p, mp.mpf("0.039141902002175105550567413579210298929290199674550951"))

    def test_periodic_orbit_error_circular(self):
        billiard50 = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 50)
        err_phi, err_p = general_lib.periodic_orbit_error_circular(billiard50, 3, 0.1, 0.6, 2 * mp.pi,
                                                                   0)
        self.assertEqual(err_phi, mp.mpf('3.8800092720241094018650736668046234310918474382909139'))
        self.assertEqual(err_p, mp.mpf("0.039141902002175105550567413579210298929290199674550951"))

        err_phi, err_p = general_lib.periodic_orbit_error_circular(billiard50, 3, 0.1, 0.6, 2 * mp.pi,
                                                                   1)
        self.assertEqual(err_phi, mp.mpf('-2.4031760351554770750602130997543823373024913604592998'))
        self.assertEqual(err_p, mp.mpf("0.039141902002175105550567413579210298929290199674550951"))

        err_phi, err_p = general_lib.periodic_orbit_error_circular(billiard50, 10, 0.1, 0.6, 2 * mp.pi,
                                                                   0)
        self.assertEqual(err_phi, mp.mpf('12.244726419176570373416190611458345715551985440064777'))
        self.assertEqual(err_p, mp.mpf('0.044453603273363039062761741490999387868689280402053229'))

        err_phi, err_p = general_lib.periodic_orbit_error_circular(billiard50, 10, 0.1, 0.6, 2 * mp.pi,
                                                                   2)
        self.assertEqual(err_phi, mp.mpf('-0.32164419518260258043438292165966582123669215743565002'))
        self.assertEqual(err_p, mp.mpf('0.044453603273363039062761741490999387868689280402053229'))

        err_phi, err_p = general_lib.periodic_orbit_error_circular(billiard50, 9, 1.6, 0.75, 2 * mp.pi,
                                                                   0)
        self.assertEqual(err_phi, mp.mpf('11.027368433012552167075546930771659035840520685686568'))
        self.assertEqual(err_p, mp.mpf('-0.17669894227663905722202679086411725100084500249986232'))

    def test_periodic_in_phi_only_orbit(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 100)
        num_iter = 22
        target_num_circles_phi = 3
        initial_phi = mp.mpf("0.1")
        p_lower = mp.mpf("0.67969")
        p_upper = mp.mpf("0.679692")
        period_in_phi = 2 * mp.pi
        p, increase_in_p = general_lib.periodic_in_phi_only_orbit(billiard, num_iter, initial_phi, p_lower, p_upper,
                                                                  period_in_phi, target_num_circles_phi)
        self.assertAlmostEqual(
            p,
            mp.mpf(
                "0.67969000002666106218039482430018775396803547530334393841822655216024406453751588042435716056277392"),
            97)
        self.assertAlmostEqual(
            increase_in_p,
            mp.mpf(
                "-0.00000000000000000000000692755647675746275910746688725978719014221884634385758206892301214560372106"),
            74)

    def test_compare_directories(self):
        temp_dir = "temp_dir_block_test"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if os.listdir(temp_dir):  # if the directory is not empty, clear it:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        dir1 = os.path.join(temp_dir, "t1")
        dir2 = os.path.join(temp_dir, "t2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        self.assertEqual(general_lib.compare_directories(dir1, dir1), (True, ""))
        file11 = os.path.join(dir1, "t1")
        with open(file11, "w") as file:
            file.write("1234")
        self.assertEqual(general_lib.compare_directories(dir1, dir2), (False, "left only"))
        file21 = os.path.join(dir2, "t1")
        with open(file21, "w") as file:
            file.write("1234")
        self.assertEqual(general_lib.compare_directories(dir1, dir2), (True, ""))
        with open(file21, "w") as file:
            file.write("124")
        self.assertEqual(general_lib.compare_directories(dir1, dir2), (False, "difference in t1"))
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        os.makedirs(dir1)
        os.makedirs(dir2)
        file11 = os.path.join(dir1, "t1")
        with open(file11, "w") as file:
            file.write("1234")
        self.assertEqual(general_lib.compare_directories(dir1, dir2), (False, "left only"))
        file21 = os.path.join(dir2, "t1")
        with open(file21, "w") as file:
            file.write("1234")
        file22 = os.path.join(dir2, "t2")
        with open(file22, "w") as file:
            file.write("dufu")
        self.assertEqual(general_lib.compare_directories(dir1, dir2), (False, "right only"))
        file12 = os.path.join(dir1, "t2")
        with open(file12, "w") as file:
            file.write("dufu")
        self.assertEqual(general_lib.compare_directories(dir1, dir2), (True, ""))
        with open(file12, "w") as file:
            file.write("dufdfjlku")
        self.assertEqual(general_lib.compare_directories(dir1, dir2), (False, "difference in t2"))
        shutil.rmtree(temp_dir)

    def test_cross2d(self):
        self.assertEqual(general_lib.cross2d([1, 0], [0, 1]), 1)
        self.assertEqual(general_lib.cross2d((1, 1), [0, 1]), 1)
        self.assertEqual(general_lib.cross2d((mp.mpf(1), mp.mpf(1)), [0, mp.mpf(1)]), 1)
        self.assertEqual(general_lib.cross2d(np.array((1, 1)), [0, 1]), 1)
        self.assertEqual(general_lib.cross2d(np.array((1, 1)), np.array([0, mp.mpf(1)])), 1)

    def test_check_plain_point_inside_angle(self):
        angle = general_lib.Angle((1, 0), (-1, 0), (0, 1))
        self.assertTrue(angle.check_plain_point_inside_angle((0, 1)))
        angle = general_lib.Angle((1, 0), (-1,-1), (0, 1))
        self.assertTrue(angle.check_plain_point_inside_angle((0, 1)))
        self.assertFalse(angle.check_plain_point_inside_angle((0, -2)))
        self.assertFalse(angle.check_plain_point_inside_angle((2, 0)))
        self.assertFalse(angle.check_plain_point_inside_angle((2, 1)))
        angle = general_lib.Angle((1, 0), (mp.mpf(-1),-1), (0, 1))
        self.assertFalse(angle.check_plain_point_inside_angle((mp.mpf(2), 0)))
        angle = general_lib.Angle((1, 0), np.array((-1,-1)), (0, 1))
        self.assertFalse(angle.check_plain_point_inside_angle(np.array((2, 1))))
        angle = general_lib.Angle((1, 0), np.array((mp.mpf(-1),-1)), (mp.mpf(0), 1))
        self.assertFalse(angle.check_plain_point_inside_angle(np.array((mp.mpf(2), 1))))

    def test_check_plain_point_in_half_plane(self):
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), 0)
        self.assertTrue(half_plane.check_plain_point_in_half_plane((1, 1)))
        half_plane = general_lib.HalfPlane(
            (mp.mpf(1), 0), (1, mp.mpf(-1)), (mp.mpf(0), 0), mp.mpf(0))
        self.assertTrue(half_plane.check_plain_point_in_half_plane((1, 1)))
        half_plane = general_lib.HalfPlane(
            np.array((mp.mpf(1), 0)), np.array((1, mp.mpf(-1))), np.array((mp.mpf(0), 0)), mp.mpf(0))
        self.assertTrue(half_plane.check_plain_point_in_half_plane((1, 1)))
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), 1)
        self.assertTrue(half_plane.check_plain_point_in_half_plane((1, 1)))
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), -1)
        self.assertTrue(half_plane.check_plain_point_in_half_plane((1, 1)))
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), -1.1)
        self.assertFalse(half_plane.check_plain_point_in_half_plane((1, 1)))
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), 0)
        self.assertTrue(half_plane.check_plain_point_in_half_plane((0, 1)))
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), 0)
        self.assertFalse(half_plane.check_plain_point_in_half_plane((0, 0.9)))
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), 0.1)
        self.assertTrue(half_plane.check_plain_point_in_half_plane((0, 0.9)))
        half_plane = general_lib.HalfPlane(
            (1, 0), (1, -1), (0, 0), 0.1)
        self.assertFalse(half_plane.check_plain_point_in_half_plane((0, 0.89)))

    def test_condition_fn_for_separatrix(self):
        condition_class = general_lib._ConditionFindingSeparatrix(
            (0, 1), (10, 1), (-10, 1), 0.1, 3)
        self.assertEqual(condition_class.condition_fn_for_separatrix((2, 3)), 1)
        self.assertEqual(condition_class.condition_fn_for_separatrix((1, 1)), -1)
        condition_class = general_lib._ConditionFindingSeparatrix(
            (0, 1), (10, 1), (-10, 1), 0.1, 1)
        self.assertIsNone(condition_class.condition_fn_for_separatrix((2, 3)))
        self.assertIsNone(condition_class.condition_fn_for_separatrix((1, 1.1)))
        self.assertEqual(condition_class.condition_fn_for_separatrix((-1, 1)), 0)
        self.assertEqual(condition_class.condition_fn_for_separatrix((-1, 1.1)), 1)
        self.assertEqual(condition_class.condition_fn_for_separatrix((0, 0)), 0)
        self.assertEqual(condition_class.condition_fn_for_separatrix((-1, 0)), 0)
        self.assertEqual(condition_class.condition_fn_for_separatrix((0.1, 0)), -1)
        self.assertEqual(condition_class.condition_fn_for_separatrix((0.1, 1)), 0)
        self.assertEqual(condition_class.condition_fn_for_separatrix((0.11, 1)), -1)
        self.assertEqual(condition_class.condition_fn_for_separatrix((0, 1.099999999)), 0)
        self.assertEqual(condition_class.condition_fn_for_separatrix((0, 1.11)), 1)

    def test_orbit_towards_hyperbolic_point(self):
        # Try towards the periodic orbit of order 3:
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 210)
        # --need the precision of 210 decimals to get to 100 near the hyperbolic point
        period = 3
        hyperbolic_point = general_lib.unpickle("../orbits/k3A0.5/periodic_orbit_3_prec100.pkl")[0, 1:]
        print(f"hyperbolic point: {hyperbolic_point}")
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
        precision = 10 ** (-100)
        separating_coefficient = 1
        forward = True  # the eigenvalue is < 1 - the direction is attracting
        starting_point = hyperbolic_point + 1e-1 * approach_separatrix_direction
        print(f"starting point: {starting_point}")
        iteration_limit = 1000
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        orbit_code, orbit = searcher.orbit_towards_hyperbolic_point(starting_point)
        self.assertEqual(orbit_code, -1)
        expected = np.array(
            [[0,
              mp.mpf(
                  '0.0921302545833015729691016427055385804977113722931164205487824542286377061376497199867689381323134932978631643694193087412713794037699699401855468749999999999999999999999999999999999999999999999999999999999999999951'),
              mp.mpf(
                  '0.529375773701616488148534858274932902787089427892518837331176266123538798010417613894899223760441837263740765834230659732260999538874494852375908910186915582147423703837021626967744677644590483406653963493581573755')],
             [1,
                mp.mpf(
                    '2.09478236910703542806465512871114497778942452558820856570277895567282639566732691678573952694088193549562930321726069782612899144666048392145039118072111221941101262608537962135248286622924526040660668056955336341'),
                mp.mpf(
                    '0.483279376512566467275979370961603124624609016554791735624755949526285496006391966330542972514359629140791142862160397368753530697820942022338377938485186477650334502499621824544591547156262452299381328269563905044')],
             [2,
                mp.mpf(
                    '-2.1830551651257424989367637114323305237618385300870457032584424107215557445022148147055225116065071645709712810699986847202897166682022184980331544675840649105695609123915718524428077613061917945564097852453645172'),
                mp.mpf(
                    '0.527304030349688187916675440579368106244549370183759269532333394415123447547521865354539742147148368665032845881705667121898193938765062668213344561317951268172104418181642534497919567163677245318683218774224388509')],
             [3,
                mp.mpf(
                    '-0.747100240924978888874557564650611276362775796687329206940105951115098328461890946203109163483436305985219277046689945818236011146545393798101355858456552963192674884338350159449569155850490423601841638338966428476'),
                mp.mpf(
                    '0.814681595520636053479963044022778629783054450506038882290335993864071583896079879638225024563194311552755326246387627976620174321981968367253390459724425280392854868103788187821087481964938717529405562622469411533')]])
        np.testing.assert_array_equal(expected, orbit)
        separating_coefficient = 1e-5
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        orbit_code, orbit = searcher.orbit_towards_hyperbolic_point(starting_point)
        self.assertEqual(orbit_code, "behind")
        np.testing.assert_array_equal(expected, orbit)

        starting_point -= np.array([0, 0.01])
        print(f"starting_point: {starting_point}")
        separating_coefficient = 1
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        orbit_code, orbit = searcher.orbit_towards_hyperbolic_point(starting_point)
        self.assertEqual(orbit_code, 1)
        expected = np.array(
            [[0,
              mp.mpf(
                  '0.0921302545833015729691016427055385804977113722931164205487824542286377061376497199867689381323134932978631643694193087412713794037699699401855468749999999999999999999999999999999999999999999999999999999999999999951'),
              mp.mpf(
                  '0.519375773701616487940368041157716051457658490125490028737426266123538798010417613894899223760441837263740765834230659732260999538874494852375908910186915582147423703837021626967744677644590483406653963493581573755')],
             [1,
              mp.mpf(
                  '2.16181070545585813624707978176387856905211671647530215909986805447944972990717409634817930554159657477552406760896175986960181078969251112989497076060085459558806070986176493932452092054160975456792799919942630453'),
              mp.mpf(
                  '0.438436002903602374797435817331680377619025511460656455500282295756510304476861913244702248642577916426571055085331444135333537155282784781503207671562348665449784154882825670236842386120357402224136899810153159682')],
             [2,
              mp.mpf(
                  '-1.6420450454258391938117554453949760391786860616676539365844562806360040017531657742631029116921385081730005628930854571661291578590370809923965844861147591767354813257056705433780406447270587044010737188766536852'),
              mp.mpf(
                  '0.159847740171693292061518530981028632694482249740408561570457527138108497566516335301652035154094247328883859449659592606658518640116641615114339166622951564187994774023478259323844544565930397959171945525850683334')],
             [3,
              mp.mpf(
                  '1.40666409797836093374606617575627458711635153772971194934579343401442058066676993625315531525318444234076169393911889873103505831537356634104904820086816340925141917637876238439298436500917060721107718445674266307'),
              mp.mpf(
                  '-0.0880238308516461884020639016982958872008371877878844152274836794244152590407586376823988290069879286546731932327958729355851461514929782725262650763027381801818830218188541488603214832231090224413791892910326187877')]])
        np.testing.assert_array_equal(expected, orbit)

        # do the same along the other_separatrix_direction, backwards:
        approach_separatrix_direction, other_separatrix_direction = (other_separatrix_direction,
                                                                     approach_separatrix_direction)
        starting_point = hyperbolic_point + 1e-3 * approach_separatrix_direction
        print(f"starting point: {starting_point}")
        separating_coefficient = 1
        forward = False
        searcher = general_lib.SearcherForSeparatrix(billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
                                                     other_separatrix_direction, precision, separating_coefficient, forward)
        orbit_code, orbit = searcher.orbit_towards_hyperbolic_point(starting_point)
        self.assertEqual(orbit_code, 1)
        expected = np.array(
            [[0,
              mp.mpf(
                  '0.00083992022231789870098961600829879982863780868755640387798571324723427390880396193844734918732762666376472380836411858739953828489888110198080539703369140625'),
              mp.mpf(
                  '0.46930360050715750630160468126989388333021035675869014017445945572691685326197987641426541832999760456863795513189994813839504670351623756333134990826430816027242370383702162696774467764459048340665396349358157371')],
             [1,
              mp.mpf(
                  '-2.094297875922690625913160727032943840388735258595750230731088401238470949067951363274249307307513542734497677651107416923349977330282144129505582495969407645754834327771219548872244341705567961732035381194885647'),
              mp.mpf(
                  '0.469784488056647420172668722927597519029488600097679664218110896630061076736689399767235736374554087617697704997520686720798634486211670139396043643053381073046083565409420816556828494498777067252006496395471526066')],
             [2,
              mp.mpf(
                  '2.09441292743983193957521306096395752039173303194301130513921461417729329101338380991931971156393940835609062252251021367874945166977177960216888450515392199442103504735296487687112956873056121740362981391749731137'),
              mp.mpf(
                  '0.469843514313976070248324844332777704091748721717480028731938537487521569969757628759784268883365050857975549980880444563366754897263992748422333714094405454016105019705746868938656218032554288093011517357681941243')],
             [3,
              mp.mpf(
                  '0.0000600423405180712570885487946826479249680109613755028515283413648663930877660146916971299688612087677131487948130715411845154068985979434315430499300092952474113339729550309749651398841590539133595840384229790376579'),
              mp.mpf(
                  '0.469883462149448544889198507343658100233089321683992968218934473415012038408289811857081304285397880258052589730410994197930242150419505050342963122177750051527319283093298994185262857990707525062222833869226660473')]])
        np.testing.assert_array_equal(expected, orbit)
        starting_point -= np.array([0, 0.01])
        searcher = general_lib.SearcherForSeparatrix(billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
                                                     other_separatrix_direction, precision, separating_coefficient, forward)
        orbit_code, orbit = searcher.orbit_towards_hyperbolic_point(starting_point)
        self.assertEqual(orbit_code, -1)
        expected = np.array(
            [[0,
              mp.mpf(
                  '0.00083992022231789870098961600829879982863780868755640387798571324723427390880396193844734918732762666376472380836411858739953828489888110198080539703369140625'),
              mp.mpf(
                  '0.45930360050715750609343786415267703200077941899166133158070945572691685326197987641426541832999760456863795513189994813839504670351623756333134990826430816027242370383702162696774467764459048340665396349358157371')],
             [1,
              mp.mpf(
                  '-2.16011377702067697796006480364687421076823994283537773898464182606607143756321409307991456902425108578807465352227915237405721779403548225885198914069750825759436187506676280729917596557682264258234394408519913364'),
              mp.mpf(
                  '0.425420013832641675848123622335219261521509344624699180883102990729264292230207301588396432025539943510860046932028427826321225876584953747039134528119498358251317718701669743995720207398757200050802657289429818315')],
             [2,
              mp.mpf(
                  '1.58128132932091227636034067393556414246505203423420213896407755471055685199179998630642984094245171113631005351262909023026938965804888174186195675735508416005527202390972134474012960052950634507034547121814784493'),
              mp.mpf(
                  '0.115530863165870073896023738655872502154635520398707377713629414701894298304843810477061165119457854268805379933101956476929936616867945349898134571541580327235343527458778967292718498775309231816099738945193439109')],
             [3,
              mp.mpf(
                  '-1.44841051670085879166688344401209461924712120906051126769636985499708748182732657747580025927614105471377780946135746902261668556085918977156463180942512074400415299806760197143547772941397805278129416031318269191'),
              mp.mpf(
                  '-0.0295822152491652564037311304083478534147944432862114060209044263622624030029752491117781780849534841064464598468124494436679421673798270061894938532215712902890009062247865759324555142034718791232995823044004266426')]])
        np.testing.assert_array_equal(expected, orbit)

    def test_orbit_hit_hyperbolic_point(self):
        # Try towards the periodic orbit of order 3:
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 210)
        # --need the precision of 210 decimals to get to 100 near the hyperbolic point
        period = 3
        hyperbolic_point = general_lib.unpickle("../orbits/k3A0.5/periodic_orbit_3_prec100.pkl")[0, 1:]
        print(f"hyperbolic point: {hyperbolic_point}")
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
        precision = 10 ** (-100)
        separating_coefficient = 1
        forward = True  # the eigenvalue is < 1 - the direction is attracting
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction
        print(f"starting point1: {starting_point1}")
        iteration_limit = 1000
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        message = None
        try:
            searcher.orbit_hit_hyperbolic_point(starting_point1, starting_point1)
        except ValueError as e:
            message = str(e)
        self.assertEqual(
            message,
            'The orbits that start at the given starting points, pass at the same side of the hyperbolic: -1')
        starting_point2 = starting_point1 - np.array([0, 0.01])
        print(f"starting point2: {starting_point2}")
        searcher_behind = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, 1e-5, forward)
        message = None
        try:
            searcher_behind.orbit_hit_hyperbolic_point(starting_point1, starting_point2)
        except RuntimeError as e:
            message = str(e)
        self.assertEqual(message,"An orbit turned out to be behind the hyperbolic point.")
        # orbit = searcher.orbit_hit_hyperbolic_point(starting_point1, starting_point2, verbose=True)  # 3 min
        # print(orbit)
        # This orbit contains 108 points (36 of them approaching each of the three periodic points)
        # and the search continues until the length of the interval gets to 1.5e-200.

        # reduce precision to make the orbit short:
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 7)
        precision = 10 ** (-10)
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        message = None
        try:
            searcher.orbit_hit_hyperbolic_point(starting_point1, starting_point2)
        except RuntimeError as e:
            message = str(e)
        self.assertEqual(
            message,
            """Binary search ended at the current dps = 7 at
phi1, p1 = [mpf('0.09213025495') mpf('0.5278874412')], -1
phi2, p2 = [mpf('0.09213025495') mpf('0.5278874338')], 1
""")
        precision = 10 ** (-3)
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        orbit = searcher.orbit_hit_hyperbolic_point(starting_point1, starting_point2)
        self.assertTrue(np.array_equal(
            orbit, np.array([[0, mp.mpf('0.09213025495'), mp.mpf('0.5278865099')],
                             [1, mp.mpf('2.104893297'), mp.mpf('0.4766033106')],
                             [2, mp.mpf('-2.093136281'), mp.mpf('0.4705879092')],
                             [3, mp.mpf('0.0006201267242'), mp.mpf('0.4696241543')]])))

        # try another direction with forward = False:
        approach_separatrix_direction, other_separatrix_direction = (
            other_separatrix_direction, approach_separatrix_direction)
        forward = False
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction
        starting_point2 = starting_point1 - np.array([0, 0.01])
        orbit = searcher.orbit_hit_hyperbolic_point(starting_point1, starting_point2, verbose=True)
        self.assertTrue(np.array_equal(
            orbit, np.array([[0, mp.mpf('0.08399202209'), mp.mpf('0.4145059809')],
                             [1, mp.mpf('-2.084639758'), mp.mpf('0.4635315426')],
                             [2, mp.mpf('2.095543236'), mp.mpf('0.4691391326')],
                             [3, mp.mpf('0.0003639757633'), mp.mpf('0.4699150845')]])))

    def test_fill_separatrix(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 7)
        period = 3
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
        separating_coefficient = 1
        forward = True  # the eigenvalue is < 1 - the direction is attracting
        precision = 10 ** (-3)
        iteration_limit = 1000
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction + np.array([0, 0.01])
        starting_point2 = starting_point1 - np.array([0, 0.02])
        separatrix1, next_starting_point1, next_starting_point2 = searcher.fill_separatrix(
            None, starting_point1, starting_point2, 0.1, 2, 0)
        self.assertTrue(np.array_equal(
            separatrix1,
            np.array([[[0, 0], [mp.mpf('0.09213025495'), mp.mpf('0.009213025449')], [mp.mpf('0.5278865173'),
                                                                                     mp.mpf('0.4757855833')]],
                      [[1, 1], [mp.mpf('2.104893178'), mp.mpf('2.095450133')], [mp.mpf('0.4766033553'),
                                                                                mp.mpf('0.4705303051')]],
                      [[2, 2], [mp.mpf('-2.093136877'), mp.mpf('-2.094290465')], [mp.mpf('0.4705882668'),
                                                                                  mp.mpf('0.4699355252')]],
                      [[3, 3], [mp.mpf('0.0006154179573'), mp.mpf('-0.0001315772533')],
                       [mp.mpf('0.4696272016'), mp.mpf('0.4699493237')]]])))
        np.testing.assert_array_almost_equal(
            next_starting_point1, hyperbolic_point * 0.99 + starting_point1 * 0.01, 7)
        np.testing.assert_array_almost_equal(
            next_starting_point2, hyperbolic_point * 0.99 + starting_point2 * 0.01, 7)
        # The following code takes 43 sec to get to the hyperbolic point within 40 decimals:
        # billiard_40 = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 90)
        # precision = 10 ** (-40)
        # searcher_40 = general_lib.SearcherForSeparatrix(
        #     billiard_40, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
        #     other_separatrix_direction, precision, separating_coefficient, forward)
        # separatrix_40, _, _ = searcher_40.fill_separatrix(
        #     None, starting_point1, starting_point2, 0.1, 2, 0)
        # print(np.array_repr(separatrix_40))
        separatrix2, next_starting_point1, next_starting_point2 = searcher.fill_separatrix(
            separatrix1, starting_point1, starting_point2, 0.1, 2, 0)
        self.assertTrue(np.array_equal(separatrix2, np.concat((separatrix1, separatrix1), axis=-1)))
        np.testing.assert_array_almost_equal(
            next_starting_point1, hyperbolic_point * 0.99 + starting_point1 * 0.01, 7)
        np.testing.assert_array_almost_equal(
            next_starting_point2, hyperbolic_point * 0.99 + starting_point2 * 0.01, 7)
        # try existing_separatrix with longer orbits:
        double_separatrix1 = np.concat((separatrix1, separatrix1))
        separatrix2, next_starting_point1, next_starting_point2 = searcher.fill_separatrix(
            double_separatrix1, starting_point1, starting_point2, 0.1, 2, 0)
        self.assertTrue(np.array_equal(separatrix2, np.concat(
            (double_separatrix1, np.pad(separatrix1, ((0, 4), (0, 0), (0, 0)), "edge")), axis=-1)))
        np.testing.assert_array_almost_equal(
            next_starting_point1, hyperbolic_point * 0.99 + starting_point1 * 0.01, 7)
        np.testing.assert_array_almost_equal(
            next_starting_point2, hyperbolic_point * 0.99 + starting_point2 * 0.01, 7)
        # try orbits of different lengths:
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 16)
        precision = 10 ** (-6)
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction + np.array([0, 0.01])
        starting_point2 = starting_point1 - np.array([0, 0.02])
        separatrix3, next_starting_point1, next_starting_point2 = searcher.fill_separatrix(
            None, starting_point1, starting_point2, 0.001, 2, 0)
        self.assertTrue(np.array_equal(
            separatrix3,
            np.array([[[0, 0],
                       [mp.mpf('0.09213025495409966036'),
                        mp.mpf('0.00009213025495409966209')],
                       [mp.mpf('0.5278874344062067514'), mp.mpf('0.4699058398561559574')]],
                      [[1, 1],
                       [mp.mpf('2.104886993554374475'), mp.mpf('2.094405672894385506')],
                       [mp.mpf('0.4766074711655671897'), mp.mpf('0.4698531525057964564')]],
                      [[2, 2],
                       [mp.mpf('-2.09319164283704795'), mp.mpf('-2.094393968716024124')],
                       [mp.mpf('0.4706236827335776168'), mp.mpf('0.4698471479250132463')]],
                      [[3, 3],
                       [mp.mpf('0.000138180913612395706'),
                        mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4699355925910207524'), mp.mpf('0.46984685777402347')]],
                      [[4, 3],
                       [mp.mpf('2.094410967104444743'), mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4698565653219421781'), mp.mpf('0.46984685777402347')]],
                      [[5, 3],
                       [mp.mpf('-2.09439330765010534'), mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4698475055250844121'), mp.mpf('0.46984685777402347')]],
                      [[6, 3],
                       [mp.mpf('-2.97577406049853721e-8'),
                        mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4698466000347480825'), mp.mpf('0.46984685777402347')]]])))
        np.testing.assert_array_almost_equal(
            next_starting_point1, hyperbolic_point * 0.999999 + starting_point1 * 1e-6, 7)
        np.testing.assert_array_almost_equal(
            next_starting_point2, hyperbolic_point * 0.999999 + starting_point2 * 1e-6, 7)
        separatrix4, next_starting_point1, next_starting_point2 = searcher.fill_separatrix(
            separatrix1, starting_point1, starting_point2, 0.001, 2, 0)
        self.assertTrue(np.array_equal(
            separatrix4,
            np.array([[[0, 0, 0, 0],
                       [mp.mpf('0.09213025495409965515'), mp.mpf('0.009213025448843836784'),
                        mp.mpf('0.09213025495409966036'),
                        mp.mpf('0.00009213025495409966209')],
                       [mp.mpf('0.527886517345905304'), mp.mpf('0.4757855832576751709'),
                        mp.mpf('0.5278874344062067514'), mp.mpf('0.4699058398561559574')]],
                      [[1, 1, 1, 1],
                       [mp.mpf('2.10489317774772644'), mp.mpf('2.095450133085250854'),
                        mp.mpf('2.104886993554374475'), mp.mpf('2.094405672894385506')],
                       [mp.mpf('0.4766033552587032318'), mp.mpf('0.4705303050577640533'),
                        mp.mpf('0.4766074711655671897'), mp.mpf('0.4698531525057964564')]],
                      [[2, 2, 2, 2],
                       [mp.mpf('-2.093136876821517944'), mp.mpf('-2.094290465116500854'),
                        mp.mpf('-2.09319164283704795'), mp.mpf('-2.094393968716024124')],
                       [mp.mpf('0.4705882668495178223'), mp.mpf('0.4699355252087116241'),
                        mp.mpf('0.4706236827335776168'), mp.mpf('0.4698471479250132463')]],
                      [[3, 3, 3, 3],
                       [mp.mpf('0.0006154179573059082031'),
                        mp.mpf('-0.0001315772533416748047'),
                        mp.mpf('0.000138180913612395706'),
                        mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4696272015571594238'), mp.mpf('0.4699493236839771271'),
                        mp.mpf('0.4699355925910207524'), mp.mpf('0.46984685777402347')]],
                      [[3, 3, 4, 3],
                       [mp.mpf('0.0006154179573059082031'),
                        mp.mpf('-0.0001315772533416748047'), mp.mpf('2.094410967104444743'),
                        mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4696272015571594238'), mp.mpf('0.4699493236839771271'),
                        mp.mpf('0.4698565653219421781'), mp.mpf('0.46984685777402347')]],
                      [[3, 3, 5, 3],
                       [mp.mpf('0.0006154179573059082031'),
                        mp.mpf('-0.0001315772533416748047'), mp.mpf('-2.09439330765010534'),
                        mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4696272015571594238'), mp.mpf('0.4699493236839771271'),
                        mp.mpf('0.4698475055250844121'), mp.mpf('0.46984685777402347')]],
                      [[3, 3, 6, 3],
                       [mp.mpf('0.0006154179573059082031'),
                        mp.mpf('-0.0001315772533416748047'),
                        mp.mpf('-2.97577406049853721e-8'),
                        mp.mpf('-5.681137330570962263e-7')],
                       [mp.mpf('0.4696272015571594238'), mp.mpf('0.4699493236839771271'),
                        mp.mpf('0.4698466000347480825'), mp.mpf('0.46984685777402347')]]])))
        np.testing.assert_array_almost_equal(
            next_starting_point1, hyperbolic_point * 0.999999 + starting_point1 * 1e-6, 7)
        np.testing.assert_array_almost_equal(
            next_starting_point2, hyperbolic_point * 0.999999 + starting_point2 * 1e-6, 7)
        separatrix5, next_starting_point1, next_starting_point2 = searcher.fill_separatrix(
            general_lib.extend_separatrix(billiard, forward, separatrix1, 1),
            starting_point1, starting_point2, 0.001, 2, 1)
        self.assertTrue(np.array_equal(separatrix5, general_lib.extend_separatrix(
            billiard, forward, separatrix4, 1)))
        np.testing.assert_array_almost_equal(
            next_starting_point1, hyperbolic_point * 0.999999 + starting_point1 * 1e-6, 7)
        np.testing.assert_array_almost_equal(
            next_starting_point2, hyperbolic_point * 0.999999 + starting_point2 * 1e-6, 7)


    def test_extend_separatrix(self):
        billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 7)
        period = 3
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
        separating_coefficient = 1
        forward = True  # the eigenvalue is < 1 - the direction is attracting
        precision = 10 ** (-3)
        iteration_limit = 1000
        searcher = general_lib.SearcherForSeparatrix(
            billiard, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        # The following has been taken from test_fill_separatrix(.):
        separatrix_to_extend = np.array(
            [[[0, 0], [mp.mpf('0.09213025495'), mp.mpf('0.009213025449')], [mp.mpf('0.5278865173'),
                                                                            mp.mpf('0.4757855833')]],
             [[1, 1], [mp.mpf('2.104893178'), mp.mpf('2.095450133')], [mp.mpf('0.4766033553'),
                                                                       mp.mpf('0.4705303051')]],
             [[2, 2], [mp.mpf('-2.093136877'), mp.mpf('-2.094290465')], [mp.mpf('0.4705882668'),
                                                                         mp.mpf('0.4699355252')]],
             [[3, 3], [mp.mpf('0.0006154179573'), mp.mpf('-0.0001315772533')],
              [mp.mpf('0.4696272016'), mp.mpf('0.4699493237')]]])
        extended_separatrix = general_lib.extend_separatrix(billiard, forward, separatrix_to_extend, 3)
        for i in range(extended_separatrix.shape[2]):
            orbit = extended_separatrix[:, :, i]
            orbit[:, 0] += 3
            orbit = billiard.set_orbit_around_point(orbit, searcher.ending_condition.singular_point)
            expected = general_lib.calculate_orbit(billiard, orbit[0, 1], orbit[0, 2], 6)
            expected = billiard.set_orbit_around_point(expected, searcher.ending_condition.singular_point)
            np.testing.assert_array_almost_equal(orbit, expected, 3)
        # change the numbering in separatrix_to_extend:
        separatrix_to_extend = np.array(
            [[[1, -1], [mp.mpf('0.09213025495'), mp.mpf('0.009213025449')], [mp.mpf('0.5278865173'),
                                                                            mp.mpf('0.4757855833')]],
             [[2, 0], [mp.mpf('2.104893178'), mp.mpf('2.095450133')], [mp.mpf('0.4766033553'),
                                                                       mp.mpf('0.4705303051')]],
             [[3, 1], [mp.mpf('-2.093136877'), mp.mpf('-2.094290465')], [mp.mpf('0.4705882668'),
                                                                         mp.mpf('0.4699355252')]],
             [[4, 2], [mp.mpf('0.0006154179573'), mp.mpf('-0.0001315772533')],
              [mp.mpf('0.4696272016'), mp.mpf('0.4699493237')]]])
        extended_separatrix = general_lib.extend_separatrix(billiard, forward, separatrix_to_extend, 3)
        for i in range(extended_separatrix.shape[2]):
            orbit = extended_separatrix[:, :, i]
            orbit[:, 0] += 2 + 2 * i
            orbit = billiard.set_orbit_around_point(orbit, searcher.ending_condition.singular_point)
            expected = general_lib.calculate_orbit(billiard, orbit[0, 1], orbit[0, 2], 6)
            expected = billiard.set_orbit_around_point(expected, searcher.ending_condition.singular_point)
            np.testing.assert_array_almost_equal(orbit, expected, 3)

    def test_create_validate_separatrix_by_blocks(self):
        temp_dir = "temp_dir_block_test"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if os.listdir(temp_dir):  # if the directory is not empty, clear it:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        filepaths = {"separatrix": f"{temp_dir}/separatrix.pkl",
                     "temp_separatrix": f"{temp_dir}/temp_separatrix.pkl",
                     "misc_outputs": f"{temp_dir}/misc_outputs.pkl",
                     "temp_misc_outputs": f"{temp_dir}/temp_misc_outputs.pkl"}
        billiard_main = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 7)
        billiard_val = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 12)
        period = 3
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
        separating_coefficient = 1
        forward = True  # the eigenvalue is < 1 - the direction is attracting
        precision = 10 ** (-3)
        iteration_limit = 1000
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction + np.array([0, 0.01])
        starting_point2 = starting_point1 - np.array([0, 0.02])
        # Check that the result is correct in case of one block:
        max_discrepancy = general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            0.1, 2, 2, 1, filepaths)
        actual1 = general_lib.unpickle(filepaths["separatrix"])
        searcher_main = general_lib.SearcherForSeparatrix(
            billiard_main, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward)
        separatrix1, _, _ = searcher_main.fill_separatrix(
            None, starting_point1, starting_point2, 0.1, 2, 1)
        np.testing.assert_array_equal(actual1, separatrix1)
        misc_outputs = general_lib.unpickle(filepaths["misc_outputs"])
        self.assertTrue(max_discrepancy < 2e-7)
        self.assertTrue(np.isnan(misc_outputs.loc[1, "discrepancy"]))
        self.assertEqual(misc_outputs.loc[0, "discrepancy"], max_discrepancy)
        shutil.rmtree(temp_dir)
        # Check that the result is correct in case of two blocks:
        max_discrepancy2 = general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            0.1, 4, 2, 1, filepaths)
        actual2 = general_lib.unpickle(filepaths["separatrix"])
        separatrix2, _, _ = searcher_main.fill_separatrix(
            None, starting_point1, starting_point2, 0.1, 4, 1)
        np.testing.assert_array_equal(actual2, separatrix2)
        misc_outputs2 = general_lib.unpickle(filepaths["misc_outputs"])
        self.assertTrue(np.isnan(misc_outputs2.loc[2, "discrepancy"]))
        self.assertTrue(1e-6 < max_discrepancy2 < 2e-6)
        shutil.rmtree(temp_dir)
        # Check that the calculations are picked up correctly if broken in the middle:
        general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            0.1, 2, 2, 1, filepaths)
        max_discrepancy3 = general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            0.1, 4, 2, 1, filepaths)
        actual3 = general_lib.unpickle(filepaths["separatrix"])
        misc_outputs3 = general_lib.unpickle(filepaths["misc_outputs"])
        self.assertEqual(max_discrepancy2, max_discrepancy3)
        np.testing.assert_array_equal(actual2, actual3)
        pd.testing.assert_frame_equal(misc_outputs2.drop(columns="time_finished"),
                                      misc_outputs3.drop(columns="time_finished"))
        shutil.rmtree(temp_dir)

    def test_extend_validate_separatrix_from_file(self):
        """Also test extend_validate_separatrix(.)"""
        temp_dir = "temp_dir_block_test"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if os.listdir(temp_dir):  # if the directory is not empty, clear it:
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        filepaths = {"separatrix": f"{temp_dir}/separatrix.pkl",
                     "temp_separatrix": f"{temp_dir}/temp_separatrix.pkl",
                     "misc_outputs": f"{temp_dir}/misc_outputs.pkl",
                     "temp_misc_outputs": f"{temp_dir}/temp_misc_outputs.pkl"}
        billiard_main = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 7)
        billiard_val = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", 12)
        period = 3
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
        separating_coefficient = 1
        forward = True  # the eigenvalue is < 1 - the direction is attracting
        precision = 10 ** (-3)
        iteration_limit = 1000
        starting_point1 = hyperbolic_point + 1e-1 * approach_separatrix_direction + np.array([0, 0.01])
        starting_point2 = starting_point1 - np.array([0, 0.02])
        # first create separatrix with num_steps = 0, and then extend it for 1 step:
        general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            0.1, 2, 2, 0, filepaths)
        logfile = f"{temp_dir}/log"
        general_lib.extend_validate_separatrix_from_file(
            billiard_main, billiard_val, forward, filepaths["separatrix"], 1, logfile)
        actual1 = general_lib.unpickle(filepaths["separatrix"])
        with open(logfile) as file:
            message = file.read()
        # message contains max_discrepancy; truncate it not to include it:
        self.assertEqual(
            message[:85], "\nExtended the separatrix away from the hyperbolic additional 1 steps, max_discrepancy")
        shutil.rmtree(temp_dir)
        # compare it with doing all in one step:
        general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            0.1, 2, 2, 1, filepaths)
        actual2 = general_lib.unpickle(filepaths["separatrix"])
        np.testing.assert_array_equal(actual2, actual1)
        shutil.rmtree(temp_dir)
        # do two steps with using alternate filepath on extending:
        general_lib.create_validate_separatrix_by_blocks(
            billiard_main, billiard_val, period, iteration_limit, hyperbolic_point, approach_separatrix_direction,
            other_separatrix_direction, precision, separating_coefficient, forward, starting_point1, starting_point2,
            0.1, 2, 2, 0, filepaths)
        new_separatrix_file = f"{temp_dir}/new.pkl"
        with open(logfile, "w") as file:
            file.write("Test.")
        max_discrepancy = general_lib.extend_validate_separatrix_from_file(
            billiard_main, billiard_val, forward, filepaths["separatrix"], 1, logfile, new_separatrix_file)
        self.assertTrue(0 < max_discrepancy < 1e-7)
        actual3 = general_lib.unpickle(new_separatrix_file)
        with open(logfile) as file:
            message = file.read()
        self.assertEqual(
            message[:90],
            "Test.\nExtended the separatrix away from the hyperbolic additional 1 steps, max_discrepancy")
        np.testing.assert_array_equal(actual3, actual2)
        shutil.rmtree(temp_dir)

    def test_remove_front_back_of_orbits(self):
        orbit = np.moveaxis(np.array([[[-1, 3, 3], [0, 2., 3.], [1, 4, 5]], [[-2, 3, 2], [-1, 2., 5], [0, 4, 6]]]),
                            0, 2)
        np.testing.assert_array_equal(general_lib.remove_front_back_of_orbits(orbit), orbit)
        np.testing.assert_array_equal(general_lib.remove_front_back_of_orbits(orbit, front=0), orbit[1:, ...])
        np.testing.assert_array_equal(general_lib.remove_front_back_of_orbits(orbit, back=0), orbit)
        np.testing.assert_array_equal(general_lib.remove_front_back_of_orbits(orbit, back=-1), orbit[:2, ...])
        np.testing.assert_array_equal(general_lib.remove_front_back_of_orbits(orbit, front=0, back=-1), orbit[1:2, ...])

    def test_flip_phi_orbit(self):
        orbit = np.array([[-1, 3, 3], [0, 2., 3.], [1, 4, 5]])
        np.testing.assert_array_equal(general_lib.flip_phi_orbit(orbit, None, None), orbit)
        np.testing.assert_array_equal(general_lib.flip_phi_orbit(orbit, 1, None),
                                      np.array([[-1, -1, 3], [0, 0., 3.], [1, -2, 5]]))
        orbit = np.array([[-1, 3, 3], [0, 2., 3.], [1, 4, 5]])
        np.testing.assert_array_equal(general_lib.flip_phi_orbit(orbit, 1, 8),
                                      np.array([[-1, 7, 3], [0, 0., 3.], [1, 6, 5]]))
        mp.mp.dps = 50
        orbit = np.moveaxis(np.array([[[-1, mp.pi, 3], [0, mp.pi / 2, 3.], [1, 3 * mp.pi / 2, 5]],
                                      [[-2, mp.pi / 2, 2], [-1, mp.pi, 5], [0, 3 * mp.pi / 2, 6]]]),
                            0, 2)
        flipped = np.moveaxis(np.array([[[-1, 0, 3], [0, mp.pi / 2, 3.], [1, 3 * mp.pi / 2, 5]],
                                        [[-2, mp.pi / 2, 2], [-1, 0, 5], [0, 3 * mp.pi / 2, 6]]]),
                              0, 2)
        np.testing.assert_array_equal(general_lib.flip_phi_orbit(orbit, mp.pi/2, 2 * mp.pi), flipped)
