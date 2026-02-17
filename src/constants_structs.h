// Copyright Leuphana Lüneburg, SSMP

// This file is part of FE-SPH-GPU

// FE-SPH-GPU is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// FE-SPH-GPU is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with FE-SPH-GPU.  If not, see <http://www.gnu.org/licenses/>.

// Copyright ETH Zurich, IWF


//This file is part of iwf_mfree_gpu_3d.

//iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

//iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.

//You should have received a copy of the GNU General Public License
//along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#ifndef CONSTANTS_STRUCTS_H_
#define CONSTANTS_STRUCTS_H_

#include "types.h"

#include <cstring>

struct phys_constants {
	float_x E;		// youngs modulus
	float_x nu;		// poisson ratio
	float_x rho0;	// reference density
	float_x K;		// bulk modulus
	float_x G;		// shear modulus
	float_x mass;	// particle mass
};

phys_constants make_phys_constants();

struct trml_constants {
	float_x cp;     // thermal capacity
	float_x tq;	    // taylor quinney constant
	float_x k;	    // thermal conductivity
	float_x alpha;	// thermal diffusitivty
	float_x T_init;	// initial temperature
	float_x eta;	// fraction of frictional power turned to heat
};

trml_constants make_trml_constants();

struct corr_constants {
	float_x wdeltap  ;	// value of kernel function at init particle spacing (for artificial stress)
	float_x stresseps;	// intensity of artificial stress
	float_x xspheps  ;	// XSPH factor (balance between interpolated and advection velocity)
	float_x alpha    ;	// artificial viscosity constant
	float_x beta     ;  // artificial viscosity constant
	float_x eta      ;  // artificial viscosity constant
};

corr_constants make_corr_constants();

//johnson cook material constants
struct joco_constants {
	float_x A;
	float_x B;
	float_x C;
	float_x n;
	float_x m;
	float_x Tmelt;
	float_x Tref;
	float_x eps_dot_ref;
	float_x clamp_temp;			//limit temperature of particles to melting temp in johnson cook?
};

joco_constants make_joco_constants();

//geometrical constants for spatial hashing
struct geom_constants {

	// number of cells in each direction
	int nx;
	int ny;
	int nz;

	// extents of spatial hashing grid
	float_x bbmin_x;
	float_x bbmin_y;
	float_x bbmin_z;

	float_x bbmax_x;
	float_x bbmax_y;
	float_x bbmax_z;

	// edge length of a cell
	float_x dx;
};

geom_constants make_geom_constants();

#endif /* CONSTANTS_STRUCTS_H_ */
