# In this file we find the values of periodic orbits of order 9 for the billiard with k = 3, A = 0.5.
import functools
import pickle
import numpy as np
import mpmath as mp
import general_lib
import billiard_Birkhoff_lib


precision_decimals = 100
billiard = billiard_Birkhoff_lib.BirkhoffBilliard_k_A(3, "0.5", precision_decimals)
period = 9
orbits_dir = "../../orbits/"


initial_values = (mp.mpf(0.249), mp.mpf(0.7424))  # Taken from the plots near this point
root = mp.findroot(functools.partial(general_lib.periodic_orbit_error, billiard, period),
                   initial_values, tol=mp.mpf(10) ** (-2 * precision_decimals + 10), maxsteps=10)
root = (root[0], root[1])
periodic_orbit = general_lib.calculate_orbit(billiard, *root, period)
print(mp.nstr(periodic_orbit, n=mp.mp.dps).replace("), (", "),\n("))
# [[0
#   mpf('0.2487964764773375876691474364240162904951365242663154249362612107912291663903924648301483953219570882595')
#   mpf('0.7424575336172858054390661980065331211403319749308098335952047574565811255775436856959868056783324202291')]
#  [1
#   mpf('1.047197551196597746154214461093167628065723133125035273658314864102605468762069666209344941780705675558')
#   mpf('0.9066943072153645694096483779683425672343006270747372686236625516768823362025920526516662838997555015679')]
#  [2
#   mpf('1.845598625915857904639281485762318965636309741983755122380368517413981771133746867588541488239454274633')
#   mpf('0.742457533617285805439066198006533121140331974930809833595204757456581125577543685695986805678332424551')]
#  [3
#   mpf('2.34319157887053307997757635861035154662658279051638597225289093899644010391453179724883827888336846699')
#   mpf('0.7424575336172858054390661980065331211403319749308098335952047574565811255775436856959868056783324202363')]
#  [4
#   mpf('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117054463')
#   mpf('0.9066943072153645694096483779683425672343006270747372686236625516768823362025920526516662838997555015536')]
#  [5
#   mpf('3.939993728309053396947710407948654221767756008233825669696998245619192708657886200007231371800865653596')
#   mpf('0.7424575336172858054390661980065331211403319749308098335952047574565811255775436856959868056783324244439')]
#  [6
#   mpf('4.437586681263728572286005280796686802758029056766456519569520667201651041438671129667528162444779845966')
#   mpf('0.7424575336172858054390661980065331211403319749308098335952047574565811255775436856959868056783324203363')]
#  [7
#   mpf('5.235987755982988730771072305465838140328615665625176368291574320513027343810348331046724708903528433612')
#   mpf('0.9066943072153645694096483779683425672343006270747372686236625516768823362025920526516662838997555015607')]
#  [8
#   mpf('6.034388830702248889256139330134989477899202274483896217013627973824403646182025532425921255362277032458')
#   mpf('0.742457533617285805439066198006533121140331974930809833595204757456581125577543685695986805678332424401')]
#  [9
#   mpf('0.2487964764773375876691474364240162904951365242663154249362612107912291663903924648301483953219570888703')
#   mpf('0.7424575336172858054390661980065331211403319749308098335952047574565811255775436856959868056783324203863')]]
filename = orbits_dir + "k3A0.5/singular_orbit_prec100.pkl"
with open(filename, "wb") as file:
    pickle.dump(periodic_orbit, file)
print(f"Created {filename}")
# Created ../orbits/k3A0.5/singular_orbit_prec100.pkl

# Find the directional derivatives and the differential similarly to study_periodic_orbits_3.py:
initial_point = periodic_orbit[1, 1:]  # at phi = pi/3
print(f"initial_point: {initial_point}")
# initial_point:
# [mpf('1.047197551196597746154214461093167628065723133125035273658314864102605468762069666209344941780705675558')
#  mpf('0.9066943072153645694096483779683425672343006270747372686236625516768823362025920526516662838997555015679')]
additions = [(direction * increment, increment) for increment in [1e-10, 1e-20, 1e-40, 1e-60]
             for direction in np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])]
directional_derivatives = [
    (general_lib.calculate_orbit(billiard, *(initial_point + addition.tolist()), period)[-1, 1:]
     - initial_point) / increment
    for addition, increment in additions]
print(f"directional_derivatives:\n{[d.astype(float) for d in directional_derivatives]}")
# directional_derivatives:
# [array([ 1.00943299e+00, -1.28052799e-04]), array([-148.02469089,    1.00943229]),
# array([-1.00943300e+00,  1.28052798e-04]), array([148.02469085,  -1.00943371]),
# array([ 1.00943300e+00, -1.28052799e-04]), array([-148.02469087,    1.009433  ]),
# array([-1.00943300e+00,  1.28052799e-04]), array([148.02469087,  -1.009433  ]),
# array([ 1.00943300e+00, -1.28052799e-04]), array([-148.02469087,    1.009433  ]),
# array([-1.00943300e+00,  1.28052799e-04]), array([148.02469087,  -1.009433  ]),
# array([ 1.00943300e+00, -1.28052799e-04]), array([-148.02469087,    1.009433  ]),
# array([-1.00943300e+00,  1.28052799e-04]), array([148.02469087,  -1.009433  ])]
matrices = [np.vstack([directional_derivatives[j] for j in [i, i+1]]).T
            for i in range(0, len(directional_derivatives), 2)]
matrices = [matrix if j % 2 == 0 else -matrix for j, matrix in enumerate(matrices)]
max_diff = max([np.abs(matrix - matrices[0]).max() for matrix in matrices])
print(f"max diff between numerical differentials: {mp.nstr(max_diff)}")
# max diff between numerical differentials: 1.41838e-6
print(np.vectorize(mp.nstr)(matrices[0]))
# [['1.00943' '-148.025']
#  ['-0.000128053' '1.00943']]
print(f"det of the differential: {np.linalg.det(matrices[0].astype(float))}")
# det of the differential: 0.9999992792336733
# similarly to study_periodic_orbits_3_contd.py, find the eigenvalues and eigenvectors of matrices[0]:
eigenvalues, eigenvectors = mp.eig(mp.matrix(matrices[0]))
print(f"the eigenvalues: {eigenvalues}")
# the eigenvalues:
# [mpf('1.147109711598199622285082369121483644682504137929052379714332311863034961431988376864060338218383212699'),
# mpf('0.8717555689075581159698735710267337656376969509836252292629587818287335590637506353334506397188701349854')]
print(f"the eigenvectors: {[eigenvectors[:, i] for i in [0, 1]]}")
# the eigenvectors: [matrix(
# [['0.9999995674638471509182553354526881563540217336072276068148802325513553888330970930555358013318820966'],
#  ['-0.0009300925322840948215269504463827921766349040246598514096769121709786889256576935976204114650650519007']]),
#  matrix(
# [['1.000001297614755439502655338055869165015729225464407144450604098488109264264878862458222973244207641'],
#  ['0.0009300989001607104954428242012033974823535864710419669426675070928071126805846830767847298499426462694']])]
# Approximately, these are [1, -0.00093] and [1, 0.00093]

# Find the directional derivatives, eigenvalues and eigenvectors for the original point with phi ~ 0.25:
initial_point = periodic_orbit[0, 1:]  # at phi = pi/3
print(f"initial_point: {initial_point}")
# initial_point:
# [mpf('0.2487964764773375876691474364240162904951365242663154249362612107912291663903924648301483953219570882595')
#  mpf('0.7424575336172858054390661980065331211403319749308098335952047574565811255775436856959868056783324202291')]
additions = [(direction * increment, increment) for increment in [1e-10, 1e-20, 1e-40, 1e-60]
             for direction in np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])]
directional_derivatives = [
    (general_lib.calculate_orbit(billiard, *(initial_point + addition.tolist()), period)[-1, 1:]
     - initial_point) / increment
    for addition, increment in additions]
print(f"directional_derivatives:\n{[d.astype(float) for d in directional_derivatives]}")
# directional_derivatives:
# [array([14.23537754,  3.62568554]), array([-48.24098827, -12.21651143]),
# array([-14.23537751,  -3.62568552]), array([48.2409885 , 12.21651163]),
# array([14.23537753,  3.62568553]), array([-48.24098838, -12.21651153]),
# array([-14.23537753,  -3.62568553]), array([48.24098838, 12.21651153]),
# array([14.23537753,  3.62568553]), array([-48.24098838, -12.21651153]),
# array([-14.23537753,  -3.62568553]), array([48.24098838, 12.21651153]),
# array([14.23537753,  3.62568553]), array([-48.24098838, -12.21651153]),
# array([-14.23537753,  -3.62568553]), array([48.24098838, 12.21651153])]
matrices = [np.vstack([directional_derivatives[j] for j in [i, i+1]]).T
            for i in range(0, len(directional_derivatives), 2)]
matrices = [matrix if j % 2 == 0 else -matrix for j, matrix in enumerate(matrices)]
max_diff = max([np.abs(matrix - matrices[0]).max() for matrix in matrices])
print(f"max diff between numerical differentials: {mp.nstr(max_diff)}")
# max diff between numerical differentials: 2.30262e-7
print(np.vectorize(mp.nstr)(matrices[0]))
# [['14.2354' '-48.241']
#  ['3.62569' '-12.2165']]
print(f"det of the differential: {np.linalg.det(matrices[0].astype(float))}")
# det of the differential: 1.0000012845088888
eigenvalues, eigenvectors = mp.eig(mp.matrix(matrices[0]))
print(f"the eigenvalues: {eigenvalues}")
# the eigenvalues:
# [mpf('1.147105870503933969979504333834047615008667407787759875341849501963515005530471572689716639477280832653'),
# mpf('0.8717602361058030922251990495845515830305052420926204079868798119711820393572609040792315466150447450958')]
# --same as for the other point in the same orbit
print(f"the eigenvectors: {[eigenvectors[:, i] for i in [0, 1]]}")
# the eigenvectors: [matrix(
# [['0.9651101519414347242999700884585499446722759736698502593685973115766687459716986108569674550981640483'],
#  ['0.2618442182282831698999220435091937883650806034226300269918620262667933202678292228394976444688632339']]),
#  matrix(
# [['0.9637200943749624086460989475440663091504021284396555594329786093427361033762647051335642194678169472'],
#  ['0.2669677175297488562583718955008840700998970845895132417689444543589460945872742465607538012258291068']])]
# find their directions:
eigenvector_directions = [float(eigenvectors[1, i] / eigenvectors[0, i]) for i in [0, 1]]
eigenvectors1 = [(1, eigenvector_direction) for eigenvector_direction in eigenvector_directions]
print(f"The eigenvectors are {eigenvectors1}")
# The eigenvectors are [(1, 0.2713101895172817), (1, 0.27701790082824357)]
