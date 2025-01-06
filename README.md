# Simulating Birkhoff billiards in Python
by Misha Byaly, Baruch Youssin
## Summary: TL;DR
This repository contains the code we created in an unsuccessful attempt to disprove the 
Birkhoff-Poritsky conjecture for a specific billiard.

Our code contains a library that can be used for simulating 
general Birkhoff billiards whose boundary is given analytically by 
its support function.

It also contains the scripts that simulate the specific billiard 
we have tried.

## Mathematical summary
Birkhoff-Poritsky conjectured that the only integrable billiards in the plane 
are ellipses. We examine a very specific billiard table which is
rotationally symmetric with respect to the angle of $\pi/3$
and has an invariant curve of 6-periodic orbits. 
Our experiments show that this billiard is not integrable 
due to complicated behaviour of separatrices near hyperbolic periodic orbits. 

We refer to the recent paper [[1]](#1) for the conjecture and further references,
and in particular, for the notion of support function that we use
to define a billiard boundary.

<a id="1">[1]</a> 
The Birkhoff-Poritsky conjecture for centrally-symmetric billiard tables.
Bialy, Misha; Mironov, Andrey E.
Ann. of Math. (2) 196 (2022), no. 1, 389–413.
[https://doi.org/10.4007/annals.2022.196.1.2](https://doi.org/10.4007/annals.2022.196.1.2)
</br>[MR4429262](https://mathscinet.ams.org/mathscinet/article?mr=4429262)