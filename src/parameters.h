/************************************************************************
 *           Brownian motion with hydrodynamic interactions             *
 ************************************************************************/

/*** Simulation parameters ***/

/* PRNG */
#define RANDOM_MAX 18446744073709551615u


#define sqrt2 1.41421356237309504880f


/* Numerical integration */
#define t0 0.0f /* Initial time */
#define tmax 0.1f /* Final time */
#define dt 0.001f /* Time step */
#define sampling 1 /* Sampling frequency */
#define PI 3.141592654f

/* System */
#define particles 1000 /* Number of particles (tested up to 250) */
#define n 3*particles /* Number of degrees of freedom */
#define gold particles /* Number of gold particles */
#define rh 1.0f /* Hydrodynamic radius */
#define D0 1.0f // 53.05 /* Diffusion coefficient */

/* Optical force parameters */
#define lambda 7.9f /* Wavelength */
#define u0 2.0 /* Laser energy density */
#define alpha 8.0 /* Polarisability (real part) */
#define beta 2.0 /* Optical dissipation factor */

/* Shear flow */
//only shear 0 in this version
#define shear 0.0

/* Confining potential */
#define boxlength 55.3 /* Simulation box side length */
#define confstrength 1.0 /* Strength */
#define confpow 4 /* Exponent (even) */
