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

//this file defines the particle class which holds the global simulation state
//	- this includes the particles contained inside the tool, if any, which are just subject to the thermal, but not mechanical solver
//	- this wastes quite a bit of memory since these particles would not need to store their stress state etc. however, handling it
//    in the way currently implemented eliminates the need to write and execute additional kernels for the heat transfer between tool
//    and work piece

#ifndef PARTICLE_GPU_H_
#define PARTICLE_GPU_H_

#include <cuda_runtime_api.h>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include "types.h"

struct particle_gpu {
	//state

	//position, velocity and velocity at boundary condition
	float4_t *pos = 0;		//waste some memory to be able to bind these to textures
	float4_t *vel = 0;
	float4_t *vel_bc = 0;

	//smoothing length, density, pressure
	float_x *h   = 0;
	float_x *rho = 0;
	float_x *p   = 0;

	//deviatoric stress state, artificial stress
	mat3x3_t *S  = 0;
	mat3x3_t *R  = 0;

	//contact forces, friction forces and normals at contact
	float3_t *fc = 0;
	float3_t *ft = 0;
	float3_t *n  = 0;

	//flags
	float_x *fixed = 0;			//particle is fixed in space
	float_x *blanked = 0;		//particle is deactivated
	float_x *tool_particle = 0;	//particle is a tool particle: no mechanical solver, thermal solver only. translated with tool velocity.


	//plastic state (equivalent)
	float_x *eps_pl;
	float_x *eps_pl_dot;

	//thermal state
	float_x  *T  = 0;

	//temporal ders
	float3_t *pos_t = 0;
	float3_t *vel_t = 0;
	float_x  *rho_t = 0;
	mat3x3_t *S_t   = 0;
	float_x  *T_t   = 0;

	//spatial ders
	mat3x3_t *S_der = 0;
	mat3x3_t *v_der = 0;

	//debug
	int *num_nbh = 0;

	//hashing
	int *idx = 0;		//unsigned int should be avoided on gpu
	int *hash = 0;		//		see best practices guide

	//count on host (!)
	unsigned int N;
	unsigned int N_init = 0;

	particle_gpu(unsigned int N);
	particle_gpu(float4_t *pos, float4_t *vel_init, float_x *rho, float_x *h, unsigned int N);
	particle_gpu(float4_t *pos, float4_t *vel_init, float_x *rho, float_x *h, float_x *fixed, unsigned int N);
	particle_gpu(float4_t *pos, float4_t *vel_init, float_x *rho, float_x *T_init, float_x *h, float_x *fixed, unsigned int N);
	particle_gpu(float4_t *pos, float4_t *vel_init, float_x *rho, float_x *T_init, float_x *h, float_x *fixed, float_x *tool_p, unsigned int N);
	particle_gpu(float4_t *pos, float4_t *vel_init, float_x *rho, float_x *h, mat3x3_t *Sini, unsigned int N);
};

#endif /* PARTICLE_GPU_H_ */
