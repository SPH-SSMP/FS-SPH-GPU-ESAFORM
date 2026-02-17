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


// This file is part of iwf_mfree_gpu_3d.

// iwf_mfree_gpu_3d is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// iwf_mfree_gpu_3d is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with mfree_iwf.  If not, see <http://www.gnu.org/licenses/>.

#include "interactions_gpu.h"

#include <thrust/device_vector.h>

#include "eigen_solver.cuh"
#include "kernels.cuh"

// physical constants on device
__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static geom_constants geometry;
__constant__ static trml_constants trml;
__constant__ static trml_constants trml_tool;
__constant__ static joco_constants johnson_cook;

// thermal constants on host
static trml_constants thermals_workpiece;
static trml_constants thermals_tool;

// is there thermal conduction in workpiece (and/or into tool?)
static bool m_thermal_workpiece = false;
static bool m_thermal_tool = false;

// texture objects for fast access to read only attributes in interactions
cudaTextureObject_t pos_tex;
cudaTextureObject_t vel_tex;
cudaTextureObject_t h_tex;
cudaTextureObject_t rho_tex;
cudaTextureObject_t p_tex;
cudaTextureObject_t T_tex;
cudaTextureObject_t tool_particle_tex;
cudaTextureObject_t fixed_tex;

// texture objects for fast access to hashing information
cudaTextureObject_t hashes_tex;
cudaTextureObject_t cells_start_tex;
cudaTextureObject_t cells_end_tex;

#ifdef USE_DOUBLE
template <typename T>
__inline__ __device__ T fetch_double(cudaTextureObject_t t, int i)
{
	int2 v = tex1Dfetch<int2>(t, i);
	return __hiloint2double(v.y, v.x);
}

/* static __inline__ __device__ double2 fetch_double(texture<int4, 1> t, int i) {
	int4 v = tex1Dfetch(t,i);
	return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
} */
template <typename T>
__inline__ __device__ T fetch_double2(cudaTextureObject_t t, int i)
{
	int4 v1 = tex1Dfetch<int4>(t, 2 * i + 0);
	int4 v2 = tex1Dfetch<int4>(t, 2 * i + 1);

	return make_double4(
		__hiloint2double(v1.y, v1.x),
		__hiloint2double(v1.w, v1.z),
		__hiloint2double(v2.y, v2.x),
		__hiloint2double(v2.w, v2.z));
}
#endif
__device__ __forceinline__ void hash(int i, int j, int k, int &idx)
{
	idx = i * geometry.ny * geometry.nz + j * geometry.nz + k;
}

__device__ __forceinline__ void unhash(int &i, int &j, int &k, int idx)
{
	i = idx / (geometry.nz * geometry.ny);
	j = (idx - i * geometry.ny * geometry.nz) / geometry.nz;
	k = idx % geometry.nz;
}

void setup_texture_objects(particle_gpu *particles, int *cells_start, int *cells_end, int num_cell)
{
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};
	texDesc.readMode = cudaReadModeElementType;

	// Setup pos_tex
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = particles->pos;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N * 4;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int4>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
#endif
	cudaCreateTextureObject(&pos_tex, &resDesc, &texDesc, nullptr);

	// Setup vel_tex
	resDesc.res.linear.devPtr = particles->vel;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N * 4;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int4>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
#endif
	cudaCreateTextureObject(&vel_tex, &resDesc, &texDesc, nullptr);

	// Setup h_tex
	resDesc.res.linear.devPtr = particles->h;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
	cudaCreateTextureObject(&h_tex, &resDesc, &texDesc, nullptr);

	// Setup rho_tex
	resDesc.res.linear.devPtr = particles->rho;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
	cudaCreateTextureObject(&rho_tex, &resDesc, &texDesc, nullptr);

	// Setup p_tex
	resDesc.res.linear.devPtr = particles->p;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
	cudaCreateTextureObject(&p_tex, &resDesc, &texDesc, nullptr);

	// Setup T_tex
	resDesc.res.linear.devPtr = particles->T;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
	cudaCreateTextureObject(&T_tex, &resDesc, &texDesc, nullptr);

	// Setup tool_particle_tex
	resDesc.res.linear.devPtr = particles->tool_particle;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
	cudaCreateTextureObject(&tool_particle_tex, &resDesc, &texDesc, nullptr);

	// Setup fixed_tex
	resDesc.res.linear.devPtr = particles->fixed;
	resDesc.res.linear.sizeInBytes = sizeof(float_x) * particles->N;
#ifdef USE_DOUBLE
	resDesc.res.linear.desc = cudaCreateChannelDesc<int2>();
#else
	resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
#endif
	cudaCreateTextureObject(&fixed_tex, &resDesc, &texDesc, nullptr);

	// Setup hashes_tex
	resDesc.res.linear.devPtr = particles->hash;
	resDesc.res.linear.sizeInBytes = sizeof(int) * particles->N;
	resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
	cudaCreateTextureObject(&hashes_tex, &resDesc, &texDesc, nullptr);

	// Setup cells_start_tex
	resDesc.res.linear.devPtr = cells_start;
	resDesc.res.linear.sizeInBytes = sizeof(int) * num_cell;
	resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
	cudaCreateTextureObject(&cells_start_tex, &resDesc, &texDesc, nullptr);

	// Setup cells_end_tex
	resDesc.res.linear.devPtr = cells_end;
	resDesc.res.linear.sizeInBytes = sizeof(int) * num_cell;
	resDesc.res.linear.desc = cudaCreateChannelDesc<int>();
	cudaCreateTextureObject(&cells_end_tex, &resDesc, &texDesc, nullptr);
}

void cleanup_texture_objects()
{
	cudaDestroyTextureObject(pos_tex);
	cudaDestroyTextureObject(vel_tex);
	cudaDestroyTextureObject(h_tex);
	cudaDestroyTextureObject(rho_tex);
	cudaDestroyTextureObject(p_tex);
	cudaDestroyTextureObject(T_tex);
	cudaDestroyTextureObject(tool_particle_tex);
	cudaDestroyTextureObject(fixed_tex);
	cudaDestroyTextureObject(hashes_tex);
	cudaDestroyTextureObject(cells_start_tex);
	cudaDestroyTextureObject(cells_end_tex);
}

__global__ void do_interactions_heat(cudaTextureObject_t pos_tex, cudaTextureObject_t h_tex, cudaTextureObject_t T_tex,
									 cudaTextureObject_t tool_particle_tex, cudaTextureObject_t hashes_tex,
									 cudaTextureObject_t cells_start_tex, cudaTextureObject_t cells_end_tex,
									 cudaTextureObject_t rho_tex, float_x *T_t, int N, float_x alpha_wp, float_x alpha_tool, float_x dz)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// load particle data at pidx
	float4_t pi = texfetch4<float4_t>(pos_tex, pidx);
	float_x hi = texfetch1<float_x>(h_tex, pidx);
	float_x Ti = texfetch1<float_x>(T_tex, pidx);

	float_x is_tool_particle = texfetch1<float_x>(tool_particle_tex, pidx);
	float_x alpha = (is_tool_particle == 1.) ? alpha_tool : alpha_wp;

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch<int>(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	int low_i = gi - 2 < 0 ? 0 : gi - 2;
	int low_j = gj - 2 < 0 ? 0 : gj - 2;
	int low_k = gk - 2 < 0 ? 0 : gk - 2;

	int high_i = gi + 3 > nx ? nx : gi + 3;
	int high_j = gj + 3 > ny ? ny : gj + 3;
	int high_k = gk + 3 > nz ? nz : gk + 3;

	float_x T_ti = 0.;

	for (int ii = low_i; ii < high_i; ii++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int kk = low_k; kk < high_k; kk++)
			{
				int idx;
				hash(ii, jj, kk, idx);

				int c_start = tex1Dfetch<int>(cells_start_tex, idx);
				int c_end = tex1Dfetch<int>(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{
					float4_t pj = texfetch4<float4_t>(pos_tex, iter);
					float_x Tj = texfetch1<float_x>(T_tex, iter);
					float_x rhoj = texfetch1<float_x>(rho_tex, iter);

					float_x w2_pse = lapl_pse(pi, pj, hi); // Laplacian by PSE-method

					float_x mass = physics.mass;

					float_x is_tool_particle_j = texfetch1<float_x>(tool_particle_tex, iter);
					if (is_tool_particle_j == 1)
						mass = rhoj * dz * dz * dz;

					T_ti += (Tj - Ti) * w2_pse * mass / rhoj;
				}
			}
		}
	}

	T_t[pidx] = alpha * T_ti;
}

__global__ void do_interactions_heat_Brookshaw(cudaTextureObject_t pos_tex, cudaTextureObject_t h_tex, cudaTextureObject_t T_tex,
											   cudaTextureObject_t tool_particle_tex, cudaTextureObject_t hashes_tex,
											   cudaTextureObject_t cells_start_tex, cudaTextureObject_t cells_end_tex,
											   cudaTextureObject_t rho_tex, float_x *T_t, int N, float_x alpha_wp, float_x alpha_tool, float_x dz)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// load particle data at pidx
	float4_t pi = texfetch4<float4_t>(pos_tex, pidx);
	float_x hi = texfetch1<float_x>(h_tex, pidx);
	float_x Ti = texfetch1<float_x>(T_tex, pidx);

	float_x is_tool_particle = texfetch1<float_x>(tool_particle_tex, pidx);
	float_x alpha = (is_tool_particle == 1.) ? alpha_tool : alpha_wp;

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch<int>(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	int low_i = gi - 2 < 0 ? 0 : gi - 2;
	int low_j = gj - 2 < 0 ? 0 : gj - 2;
	int low_k = gk - 2 < 0 ? 0 : gk - 2;

	int high_i = gi + 3 > nx ? nx : gi + 3;
	int high_j = gj + 3 > ny ? ny : gj + 3;
	int high_k = gk + 3 > nz ? nz : gk + 3;

	float_x T_lapl = 0.;

	for (int ii = low_i; ii < high_i; ii++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int kk = low_k; kk < high_k; kk++)
			{
				int idx;
				hash(ii, jj, kk, idx);

				int c_start = tex1Dfetch<int>(cells_start_tex, idx);
				int c_end = tex1Dfetch<int>(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					if (alpha != 0.)
					{
						float4_t pj = texfetch4<float4_t>(pos_tex, iter);
						float_x Tj = texfetch1<float_x>(T_tex, iter);
						float_x rhoj = texfetch1<float_x>(rho_tex, iter);

						float4_t ww = cubic_spline(pi, pj, hi);
						float_x w = ww.x;
						float_x w_x = ww.y;
						float_x w_y = ww.z;
						float_x w_z = ww.w;

						float_x xij = pi.x - pj.x;
						float_x yij = pi.y - pj.y;
						float_x zij = pi.z - pj.z;
						float_x rij = sqrt(xij * xij + yij * yij + zij * zij);
						if (rij > 1e-8)
						{
							float_x eijx = xij / rij;
							float_x eijy = yij / rij;
							float_x eijz = zij / rij;
							float_x rij1 = 1. / rij;

							float_x mass = physics.mass;

							float_x is_tool_particle_j = texfetch1<float_x>(tool_particle_tex, iter);
							if (is_tool_particle_j == 1)
								mass = rhoj * dz * dz * dz;

							T_lapl += 2.0 * (mass / rhoj) * (Ti - Tj) * rij1 * (eijx * w_x + eijy * w_y + eijz * w_z);
						}
					}
				}
			}
		}
	}

	T_t[pidx] = alpha * T_lapl;
}

__device__ double sigma_yield_interaction_2(joco_constants jc, double eps_pl, double eps_pl_dot, double t)
{
	double theta = (t - jc.Tref) / (jc.Tmelt - jc.Tref);

	double Term_A = jc.A + jc.B * pow(eps_pl, jc.n);
	// double Term_A = jc.A +(678.e6) * pow(eps_pl, 0.71);
	double Term_B = 1.0;

	double eps_dot = eps_pl_dot / jc.eps_dot_ref;

	if (eps_dot > 1.0)
	{
		Term_B = 1.0 + jc.C * log_t(eps_dot);
	}
	else
	{
		Term_B = pow((1.0 + eps_dot), jc.C);
	}

	double Term_C = 1.0 - pow(theta, jc.m);
	return Term_A * Term_B * Term_C;
}

__device__ void kirk_contact_force(vec3_t &fN_each, float_x pd, float3_t vij, float3_t nav, float_x dt)
{
	float_x DFAC = 0.2;

	float_x PFAC_init = 0.1;

	float_x PFAC = PFAC_init;

	float_x slave_mass = physics.mass;
	float_x dpN = vij.x * nav.x + vij.y * nav.y + vij.z * nav.z;

	float_x kij = ((slave_mass * slave_mass) / (slave_mass + slave_mass)) * (PFAC / (dt * dt));
	float_x cij = DFAC * (slave_mass + slave_mass) * sqrt_t((kij * (slave_mass + slave_mass)) / (slave_mass * slave_mass));

	fN_each.x += -1. * (kij * pd + cij * dpN) * nav.x;
	fN_each.y += -1. * (kij * pd + cij * dpN) * nav.y;
	fN_each.z += -1. * (kij * pd + cij * dpN) * nav.z;
}

__global__ void do_interactions_monaghan(cudaTextureObject_t pos_tex, cudaTextureObject_t T_tex, cudaTextureObject_t vel_tex, cudaTextureObject_t h_tex,
										 cudaTextureObject_t rho_tex, cudaTextureObject_t p_tex, cudaTextureObject_t tool_particle_tex,
										 cudaTextureObject_t hashes_tex, cudaTextureObject_t cells_start_tex, cudaTextureObject_t cells_end_tex,
										 const float_x *__restrict__ blanked, const mat3x3_t *__restrict__ S, const mat3x3_t *__restrict__ R,
										 mat3x3_t *__restrict__ v_der, mat3x3_t *__restrict__ S_der, float_x *__restrict__ T_t,
										 float3_t *__restrict__ pos_t, float3_t *__restrict__ vel_t, unsigned int N, cudaTextureObject_t fixed_tex,
										 float_x dz, float_x *eps_pl, float_x *eps_pl_dot, float_x dt, float_x* temp, 
										 float4_t* vel, float_x* fixed, float3_t *n)
{
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= N)
		return;
	if (blanked[pidx] == 1.)
		return;

	float_x is_tool_particle_i = texfetch1<float_x>(tool_particle_tex, pidx);
	float_x is_fixed_i = texfetch1<float_x>(fixed_tex, pidx);

	/* if (is_fixed_i > 1.)
		return; */

	// load physical constants
	float_x mass = physics.mass;
	float_x K = physics.K;
#ifdef Thermal_Conduction_Brookshaw
	float_x thermal_alpha = (is_tool_particle_i == 0.) ? trml.alpha : trml_tool.alpha;
#endif

	// load correction constants
	float_x wdeltap = correctors.wdeltap;
	float_x alpha = correctors.alpha;
	float_x beta = correctors.beta;
	float_x eta = correctors.eta;
	float_x eps = correctors.xspheps;

	//float_x eps = (is_fixed_i == 1.) ? 0.5 : correctors.xspheps;

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;

	// load particle data at pidx
	float4_t pi = texfetch4<float4_t>(pos_tex, pidx);
	float4_t vi = texfetch4<float4_t>(vel_tex, pidx);

	mat3x3_t Si = S[pidx];
	mat3x3_t Ri = R[pidx];
	float_x hi = texfetch1<float_x>(h_tex, pidx);
	float_x rhoi = texfetch1<float_x>(rho_tex, pidx);
	float_x prsi = texfetch1<float_x>(p_tex, pidx);
	float_x Ti = texfetch1<float_x>(T_tex, pidx);
	float_x eps_pl_i = eps_pl[pidx];
	float_x eps_pl_dot_i = eps_pl_dot[pidx];

	float_x rhoi21 = 1. / (rhoi * rhoi);

	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch<int>(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	// find neighboring boxes (take care not to iterate beyond size of cell lists structure)
	int low_i = gi - 2 < 0 ? 0 : gi - 2;
	int low_j = gj - 2 < 0 ? 0 : gj - 2;
	int low_k = gk - 2 < 0 ? 0 : gk - 2;

	int high_i = gi + 3 > nx ? nx : gi + 3;
	int high_j = gj + 3 > ny ? ny : gj + 3;
	int high_k = gk + 3 > nz ? nz : gk + 3;

	// init vars to be written at pidx
	mat3x3_t vi_der(0.);
	mat3x3_t Si_der(0.);
	float3_t vi_t = make_float3_t(0., 0., 0.);
	float3_t vi_adv_t = make_float3_t(0., 0., 0.);
	float3_t xi_t = make_float3_t(0., 0., 0.);

#ifdef Thermal_Conduction_Brookshaw
	float_x T_lapl = 0.; // Laplacian of temperature field
#endif

#ifdef CSPM
	mat3x3_t B(0.);

	if (true) // only fixed 0 and 1 are allowed to interact
	{
		// iterate over neighboring boxes
		for (int ii = low_i; ii < high_i; ii++)
		{
			for (int jj = low_j; jj < high_j; jj++)
			{
				for (int kk = low_k; kk < high_k; kk++)
				{
					int idx;
					hash(ii, jj, kk, idx);

					// iterate over particles contained in a neighboring box
					int c_start = tex1Dfetch<int>(cells_start_tex, idx);
					int c_end = tex1Dfetch<int>(cells_end_tex, idx);

					if (c_start == 0xffffffff)
						continue;

					for (int iter = c_start; iter < c_end; iter++)
					{

						if (blanked[iter] == 1.)
						{
							continue;
						}

						float_x is_tool_particle_j = texfetch1<float_x>(tool_particle_tex, iter);
						float_x is_fixed_j = texfetch1<float_x>(fixed_tex, iter);
						float4_t vj = texfetch4<float4_t>(vel_tex, iter);
						float4_t pj = texfetch4<float4_t>(pos_tex, iter);

						if (is_tool_particle_i == is_tool_particle_j)
						{

							float_x rhoj = texfetch1<float_x>(rho_tex, iter);

							const float_x volj = mass / rhoj;

							float4_t ww = cubic_spline(pi, pj, hi);

							float_x w_x = ww.y;
							float_x w_y = ww.z;
							float_x w_z = ww.w;

							const float_x delta_x = pi.x - pj.x;
							const float_x delta_y = pi.y - pj.y;
							const float_x delta_z = pi.z - pj.z;

							// copute CSPM / Randles Libersky Correction Matrix
							B[0][0] -= volj * delta_x * w_x;
							B[1][0] -= volj * delta_x * w_y;
							B[2][0] -= volj * delta_x * w_z;

							B[0][1] -= volj * delta_y * w_x;
							B[1][1] -= volj * delta_y * w_y;
							B[2][1] -= volj * delta_y * w_z;

							B[0][2] -= volj * delta_z * w_x;
							B[1][2] -= volj * delta_z * w_y;
							B[2][2] -= volj * delta_z * w_z;
						}
					}
				}
			}
		}
	}

	// save invert
	mat3x3_t invB(1.);
	float_x det_B = glm::determinant(B);
	if (det_B > 1e-8)
	{
		invB = glm::inverse(B);
	}
#endif

	// iterate over neighboring boxes
	for (int ii = low_i; ii < high_i; ii++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int kk = low_k; kk < high_k; kk++)
			{
				int idx;
				hash(ii, jj, kk, idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch<int>(cells_start_tex, idx);
				int c_end = tex1Dfetch<int>(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					if (blanked[iter] == 1.)
					{
						continue;
					}
					// load vars at neighbor particle
					float4_t pj = texfetch4<float4_t>(pos_tex, iter);
					float4_t vj = texfetch4<float4_t>(vel_tex, iter);

					mat3x3_t Sj = S[iter];
					mat3x3_t Rj = R[iter];
					float_x hj = texfetch1<float_x>(h_tex, iter);
					float_x rhoj = texfetch1<float_x>(rho_tex, iter);
					float_x prsj = texfetch1<float_x>(p_tex, iter);
#ifdef Thermal_Conduction_Brookshaw
					float_x Tj = 0.;
					if (thermal_alpha != 0.)
					{
						Tj = texfetch1<float_x>(T_tex, iter);
					}
#endif
					float_x is_tool_particle_j = texfetch1<float_x>(tool_particle_tex, iter);
					float_x is_fixed_j = texfetch1<float_x>(fixed_tex, iter);

					float_x volj = mass / rhoj;
					float_x rhoj21 = 1. / (rhoj * rhoj);

					// compute kernel
					float4_t ww = cubic_spline(pi, pj, hi);

					// correct by CSPM matrix if def'd
#ifndef CSPM
					float_x w = ww.x;
					float_x w_x = ww.y;
					float_x w_y = ww.z;
					float_x w_z = ww.w;
#else
					float_x w = ww.x;
					float_x w_x = (ww.y * invB[0][0] + ww.z * invB[1][0] + ww.w * invB[2][0]);
					float_x w_y = (ww.y * invB[0][1] + ww.z * invB[1][1] + ww.w * invB[2][1]);
					float_x w_z = (ww.y * invB[0][2] + ww.z * invB[1][2] + ww.w * invB[2][2]);
#endif

					if (is_tool_particle_i == is_tool_particle_j){
						//Kernel Gradient (Color Function / Free Surface Detection by Divergence)
						//Uses the normalized gradient of the kernel sums to identify surface orientation.
						//$\mathbf{n}_i = \sum_j \frac{m_j}{\rho_j} \nabla W_{ij}$
						//Magnitude of $\mathbf{n}_i$ is near zero inside, large near the surface.
						//Thresholding $|\mathbf{n}_i|$ detects free surface.
						//Normal vector also gives surface orientation.
						//Pros: Provides smooth normals → good for rendering, boundary conditions.
						//Cons: Computationally heavier than neighbor count.
						n[pidx].x += ww.y * volj;
						n[pidx].y += ww.z * volj;
						n[pidx].z += ww.w * volj;
					}

					
					if (is_tool_particle_i == is_tool_particle_j)
					{
						// derive vel
						vi_der[0][0] += (vj.x - vi.x) * w_x * volj;
						vi_der[0][1] += (vj.x - vi.x) * w_y * volj;
						vi_der[0][2] += (vj.x - vi.x) * w_z * volj;

						vi_der[1][0] += (vj.y - vi.y) * w_x * volj;
						vi_der[1][1] += (vj.y - vi.y) * w_y * volj;
						vi_der[1][2] += (vj.y - vi.y) * w_z * volj;

						vi_der[2][0] += (vj.z - vi.z) * w_x * volj;
						vi_der[2][1] += (vj.z - vi.z) * w_y * volj;
						vi_der[2][2] += (vj.z - vi.z) * w_z * volj;

						float_x Rxx = 0.;
						float_x Ryy = 0.;
						float_x Rzz = 0.;

						float_x Rxy = 0.;
						float_x Rxz = 0.;
						float_x Ryz = 0.;

						// compute artificial stress
						if (wdeltap > 0)
						{
							float_x fab = w / wdeltap;
							fab *= fab; // to the power of 4
							fab *= fab;

							Rxx = fab * (Ri[0][0] + Rj[0][0]);
							Rxy = fab * (Ri[0][1] + Rj[0][1]);
							Ryy = fab * (Ri[1][1] + Rj[1][1]);
							Rxz = fab * (Ri[0][2] + Rj[0][2]);
							Ryz = fab * (Ri[1][2] + Rj[1][2]);
							Rzz = fab * (Ri[2][2] + Rj[2][2]);
						}

						// derive stress
						Si_der[0][0] += mass * ((Si[0][0] - prsi) * rhoi21 + (Sj[0][0] - prsj) * rhoj21 + Rxx) * w_x;
						Si_der[0][1] += mass * (Si[0][1] * rhoi21 + Sj[0][1] * rhoj21 + Rxy) * w_y;
						Si_der[0][2] += mass * (Si[0][2] * rhoi21 + Sj[0][2] * rhoj21 + Rxz) * w_z;

						Si_der[1][0] += mass * (Si[1][0] * rhoi21 + Sj[1][0] * rhoj21 + Rxy) * w_x;
						Si_der[1][1] += mass * ((Si[1][1] - prsi) * rhoi21 + (Sj[1][1] - prsj) * rhoj21 + Ryy) * w_y;
						Si_der[1][2] += mass * (Si[1][2] * rhoi21 + Sj[1][2] * rhoj21 + Ryz) * w_z;

						Si_der[2][0] += mass * (Si[2][0] * rhoi21 + Sj[2][0] * rhoj21 + Rxz) * w_x;
						Si_der[2][1] += mass * (Si[2][1] * rhoi21 + Sj[2][1] * rhoj21 + Ryz) * w_y;
						Si_der[2][2] += mass * ((Si[2][2] - prsi) * rhoi21 + (Sj[2][2] - prsj) * rhoj21 + Rzz) * w_z;

						// artificial viscosity
						float_x xij = pi.x - pj.x;
						float_x yij = pi.y - pj.y;
						float_x zij = pi.z - pj.z;

						float_x vijx = vi.x - vj.x;
						float_x vijy = vi.y - vj.y;
						float_x vijz = vi.z - vj.z;

						float_x vijposij = vijx * xij + vijy * yij + vijz * zij;
						float_x rhoij = 0.5 * (rhoi + rhoj);

						if (vijposij < 0.)
						{
							float_x ci = sqrtf(K / rhoi);
							float_x cj = sqrtf(K / rhoj);

							float_x cij = 0.5 * (ci + cj);
							float_x hij = 0.5 * (hi + hj);

							float_x r2ij = xij * xij + yij * yij + zij * zij;
							float_x muij = (hij * vijposij) / (r2ij + eta * eta * hij * hij);
							float_x piij = (-alpha * cij * muij + beta * muij * muij) / rhoij;

							vi_t.x += -mass * piij * w_x;
							vi_t.y += -mass * piij * w_y;
							vi_t.z += -mass * piij * w_z;
						}

						// add xsph correction
						xi_t.x += -eps * w * mass / rhoij * vijx;
						xi_t.y += -eps * w * mass / rhoij * vijy;
						xi_t.z += -eps * w * mass / rhoij * vijz;
					}

#ifdef Thermal_Conduction_Brookshaw
					// thermal, 3D Brookshaw
					if (thermal_alpha != 0.)
					{
						float4_t pj = texfetch4<float4_t>(pos_tex, iter);
						float_x xij = pi.x - pj.x;
						float_x yij = pi.y - pj.y;
						float_x zij = pi.z - pj.z;
						float_x rij = sqrt(xij * xij + yij * yij + zij * zij);
						if (rij > 1e-8)
						{
							float_x eijx = xij / rij;
							float_x eijy = yij / rij;
							float_x eijz = zij / rij;
							float_x rij1 = 1. / rij;
							T_lapl += 2.0 * (mass / rhoj) * (Ti - Tj) * rij1 * (eijx * w_x + eijy * w_y + eijz * w_z);
						}
					}
#endif
				}
			}
		}
	}

	// normalize normal vector and invert
	{
		float3_t ni = n[pidx];
		float_x niorm_ni = sqrt(ni.x * ni.x + ni.y * ni.y + ni.z * ni.z);

		if (niorm_ni > 1e-8)
		{
			n[pidx].x = -1. * ni.x / niorm_ni;
			n[pidx].y = -1. * ni.y / niorm_ni;
			n[pidx].z = -1. * ni.z / niorm_ni;
		}
		else
		{
			n[pidx].x = 0.;
			n[pidx].y = 0.;
			n[pidx].z = 0.;
		}
	}

	// write back
	S_der[pidx] = Si_der;
	v_der[pidx] = vi_der;

	pos_t[pidx] = xi_t;
	vel_t[pidx] = vi_t;

#ifdef Thermal_Conduction_Brookshaw
	if (thermal_alpha != 0.)
	{
		T_t[pidx] = thermal_alpha * T_lapl;
	}
#endif
}


__global__ void do_interactions_rod_force_Songwon(particle_gpu particles, float_x dt, float3_t *forces, float *fr_heat_gen,  cudaTextureObject_t hashes_tex, cudaTextureObject_t cells_start_tex, cudaTextureObject_t cells_end_tex, float_x di)
{

	// exhaustive contact algorithm (n^2)

	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int N = particles.N;
	if (pidx >= N)
		return;
	if (particles.blanked[pidx] == 1.)
		return;

	if (particles.tool_particle[pidx] != 0)
		return;

	vec3_t fN(0., 0., 0.);

	// load particle data at pidx
	float4_t vi = particles.vel[pidx];
	float_x rhoi = particles.rho[pidx];
	float_x Ci = sqrt(physics.K / rhoi);

	float_x hi = particles.h[pidx];
	float4_t pi = particles.pos[pidx];

	// load geometrical constants
	int nx = geometry.nx;
	int ny = geometry.ny;
	int nz = geometry.nz;


	// unhash and look for neighbor boxes
	int hashi = tex1Dfetch<int>(hashes_tex, pidx);
	int gi, gj, gk;
	unhash(gi, gj, gk, hashi);

	// find neighboring boxes (take care not to iterate beyond size of cell lists structure)
	int low_i = gi - 2 < 0 ? 0 : gi - 2;
	int low_j = gj - 2 < 0 ? 0 : gj - 2;
	int low_k = gk - 2 < 0 ? 0 : gk - 2;

	int high_i = gi + 3 > nx ? nx : gi + 3;
	int high_j = gj + 3 > ny ? ny : gj + 3;
	int high_k = gk + 3 > nz ? nz : gk + 3;

	float_x ni[3];
	ni[0] = particles.n[pidx].x;
	ni[1] = particles.n[pidx].y;
	ni[2] = particles.n[pidx].z;

	float_x SMALL = 1.0e-15;

	float_x slave_mass = physics.mass;

	float_x nav[3];
	nav[0] = 0;
	nav[1] = 0;
	nav[2] = 0;

	// iterate over neighboring boxes
	for (int ii = low_i; ii < high_i; ii++)
	{
		for (int jj = low_j; jj < high_j; jj++)
		{
			for (int kk = low_k; kk < high_k; kk++)
			{
				int idx;
				hash(ii, jj, kk, idx);

				// iterate over particles contained in a neighboring box
				int c_start = tex1Dfetch<int>(cells_start_tex, idx);
				int c_end = tex1Dfetch<int>(cells_end_tex, idx);

				if (c_start == 0xffffffff)
					continue;

				for (int iter = c_start; iter < c_end; iter++)
				{

					if (particles.blanked[iter] == 1. || particles.tool_particle[iter] == 0)
					{
						continue;
					}

					// load vars at neighbor particle
					float4_t pj = particles.pos[iter];
					float4_t vj = particles.vel[iter];

					float_x xij = pi.x - pj.x;
					float_x yij = pi.y - pj.y;
					float_x zij = pi.z - pj.z;

					float_x vxij = vi.x - vj.x;
					float_x vyij = vi.y - vj.y;
					float_x vzij = vi.z - vj.z;

					float_x r_2 = (xij * xij + yij * yij + zij * zij);

					if (r_2 < di*di)
					{

						float_x nj[3];
						nj[0] = particles.n[iter].x;
						nj[1] = particles.n[iter].y;
						nj[2] = particles.n[iter].z;

						float_x ab = -ni[0] * nj[0] - ni[1] * nj[1] - ni[2] * nj[2];
						float_x a_abs = sqrt(ni[0] * ni[0] + ni[1] * ni[1] + ni[2] * ni[2]);
						float_x b_abs = sqrt(nj[0] * nj[0] + nj[1] * nj[1] + nj[2] * nj[2]);

						// protect against zero-length normals and numerical drift outside [-1,1]
						float_x Theta_degree = 0.;
						float_x denom = a_abs * b_abs;
						if (denom > 1e-12)
						{
							float_x cosTheta = ab / denom;
							if (cosTheta > 1.0)
								cosTheta =1.0;
							else if (cosTheta < -1.0)
								cosTheta = -1.0;
							float_x Theta_radian = acos(cosTheta);
							Theta_degree = (Theta_radian * 180.) / 3.14159265359;
						}
						else
						{
							// one or both normals are (near) zero length, treat angle as zero
							Theta_degree = 0.;
						}

						float_x nij[3];
						float_x absnij;
						nij[0] = ni[0] - nj[0];
						nij[1] = ni[1] - nj[1];
						nij[2] = ni[2] - nj[2];
						absnij = sqrt(nij[0] * nij[0] + nij[1] * nij[1] + nij[2] * nij[2]);

						if (Theta_degree < 70) // 70 need to be obtimized
						{

							float_x test1 = xij * ni[0] + yij * ni[1] + zij * ni[2];
							float_x test2 = -xij * nj[0] - yij * nj[1] - zij * nj[2];
							float_x test3 = xij * nij[0] + yij * nij[1] + zij * nij[2];
							test3 /= (absnij + SMALL);

							if (test1 > max(test2, test3))
							{
								nav[0] = ni[0];
								nav[1] = ni[1];
								nav[2] = ni[2];
							}
							else if (test2 > max(test1, test3))
							{
								nav[0] = -nj[0];
								nav[1] = -nj[1];
								nav[2] = -nj[2];
							}
							else
							{
								nav[0] = nij[0] / (absnij + SMALL);
								nav[1] = nij[1] / (absnij + SMALL);
								nav[2] = nij[2] / (absnij + SMALL);
							}
						}
						else
						{
							nav[0] = nij[0] / (absnij + SMALL);
							nav[1] = nij[1] / (absnij + SMALL);
							nav[2] = nij[2] / (absnij + SMALL);
						}

						float_x rijDotnav = xij * nav[0] + yij * nav[1] + zij * nav[2];
						float_x pnav = di - fabs(rijDotnav);
						float_x pnavDot = vxij * nav[0] + vyij * nav[1] + vzij * nav[2];

						if (pnav > 0)
						{

							/*************************************** Prof. Songwon's contact method: doi:10.1016/j.ijimpeng.2007.04.009 *********************************/
							float_x pd = pnav;
							float_x dpN = pnavDot;
							float_x PFAC = 0.01;

							float_x rhoj = particles.rho[iter];
							float_x Cj = sqrt(physics.K / rhoj);
							float_x Ei_init = physics.E;
							float_x Ej_init = physics.E;
							float_x d0 = di;

							float_x Ei = Ei_init;
							float_x Ej = Ej_init;

							float_x alpha_1 = ((rhoj * Cj) / (rhoj * Cj + rhoi * Ci)) * (rhoi * Ci);
							float_x alpha_2;
							if (Ei == 0 || Ej == 0)
								alpha_2 = 0;
							else
								alpha_2 = ((Ei * Ej) / (Ei + Ej)) * (1. / d0);

							float_x mass_j = physics.mass;
							float_x vol_j = mass_j / rhoj;

							float_x mass_i = physics.mass;
							float_x vol_i = mass_i / rhoi;

							float_x area_i = 2 * 3.14159265359 * (di / 2.) * hi;

							float4_t ww = cubic_spline(pi, pj, hi);
							float_x w = ww.x;
							float_x w_x = ww.y;
							float_x w_y = ww.z;
							float_x w_z = ww.w;

							vec3_t fN_each(0., 0., 0.);
							float_x ff = 0.1;
							fN_each.x = -1. * ff * (alpha_1 * dpN + alpha_2 * pd) * nav[0] * area_i * vol_j * (w);
							fN_each.y = -1. * ff * (alpha_1 * dpN + alpha_2 * pd) * nav[1] * area_i * vol_j * (w);
							fN_each.z = -1. * ff * (alpha_1 * dpN + alpha_2 * pd) * nav[2] * area_i * vol_j * (w);

							fN.x += fN_each.x;
							fN.y += fN_each.y;
							fN.z += fN_each.z;
							/************************************************** Friction force ******************************************************/

							// Calculating the friction force
							vec3_t fT(0., 0., 0.);
							vec3_t vm = vec3_t(vj.x, vj.y, vj.z); // velocity of substrate particle or joined one
							vec3_t vp = vec3_t(vi.x, vi.y, vi.z); // velocity of rod particle
							vec3_t nis = vec3_t(nav[0], nav[1], nav[2]);
							vec3_t v = vp - vm;
							vec3_t vr = v - v * nis;
							{
								// initialize the friction force by zerp
								float3_t fric;
								fric.x = 0.;
								fric.y = 0.;
								fric.z = 0.;

								vec3_t fricold(fric.x, fric.y, fric.z);

								// Adding the effect of temp in decreasing the fric surface coefficient
								float_x friction_mu_init = 0.35;

								float_x TT = particles.T[pidx];
								float_x TTT = particles.T[iter];

								if (TT >= johnson_cook.Tmelt || TTT >= johnson_cook.Tmelt)
								{
									friction_mu_init = 0.0;
								}

								float_x friction_mu = friction_mu_init;

								float_x contact_alpha = alpha_2;
								vec3_t kdeltae = contact_alpha * slave_mass * vr / dt;

								float_x fy = friction_mu * glm::length(fN_each);

								vec3_t fstar = fricold - kdeltae;

								float_x fstar_mag = glm::length(fstar);
								if (fstar_mag != 0)
								{
									fT = fy * fstar / fstar_mag;
								}
							}

							particles.ft[pidx].x += fT.x;
							particles.ft[pidx].y += fT.y;
							particles.ft[pidx].z += fT.z;

							/************************************************** Heat generation ******************************************************/
							float_x f_fric_mag = sqrtf(fT.x * fT.x + fT.y * fT.y + fT.z * fT.z);

							if (f_fric_mag != 0.)
							{

								float_x v_rel_mag = sqrtf(vr.x * vr.x + vr.y * vr.y + vr.z * vr.z);

								float_x rod_T = 0.5 * trml.eta * dt * f_fric_mag * v_rel_mag / (trml.cp * physics.mass); // rod
								particles.T[pidx] += rod_T;
								float_x sub_T = 0.5 * trml_tool.eta * dt * f_fric_mag * v_rel_mag / (trml_tool.cp * physics.mass); // substrate
								atomicAdd(&(particles.T[iter]), sub_T);

								float total_T = rod_T + sub_T;
								atomicAdd(&(fr_heat_gen[0]), total_T);
							}

							atomicAdd(&(particles.fc[iter].x), (-1. * fN_each.x));
							atomicAdd(&(particles.fc[iter].y), (-1. * fN_each.y));
							atomicAdd(&(particles.fc[iter].z), (-1. * fN_each.z));

							atomicAdd(&(particles.ft[iter].x), (-1. * fT.x));
							atomicAdd(&(particles.ft[iter].y), (-1. * fT.y));
							atomicAdd(&(particles.ft[iter].z), (-1. * fT.z));
						}
					}
				}
			}
		}
	}

	particles.fc[pidx].x = fN.x;
	particles.fc[pidx].y = fN.y;
	particles.fc[pidx].z = fN.z;

	float_x fx = fN.x + particles.ft[pidx].x;
	float_x fy = fN.y + particles.ft[pidx].y;
	float_x fz = fN.z + particles.ft[pidx].z;

	fx = (isnan(fx)) ? 0. : fx;
	fy = (isnan(fy)) ? 0. : fy;
	fz = (isnan(fz)) ? 0. : fz;

	atomicAdd(&(forces[0].x), fx);
	atomicAdd(&(forces[0].y), fy);
	atomicAdd(&(forces[0].z), fz);
}

void interactions_rod_force_Songwon(particle_gpu *particles,  int *cells_start,
									 int *cells_end, int num_cell, float3_t *forces, float *fr_heat_gen)
{
	setup_texture_objects(particles, cells_start, cells_end, num_cell);

	// run all interactions in one go
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	// reset forces and heat generation
	cudaMemset(forces, 0, sizeof(float3_t));
	cudaMemset(fr_heat_gen, 0, sizeof(float));

	check_cuda_error("before interactions_rod_force_Songwon\n");

	do_interactions_rod_force_Songwon<<<dG, dB>>>(*particles, global_time_dt, forces,
		 										fr_heat_gen, hashes_tex,  cells_start_tex, 
		 										 cells_end_tex, global_dz);
	cleanup_texture_objects();

	check_cuda_error("interactions interactions_rod_force_Songwon\n");
}
void interactions_monaghan(particle_gpu *particles, int *cells_start, int *cells_end, int num_cell)
{
	setup_texture_objects(particles, cells_start, cells_end, num_cell);

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	check_cuda_error("before interactions monaghan\n");

	do_interactions_monaghan<<<dG, dB>>>(pos_tex, T_tex, vel_tex, h_tex, rho_tex, p_tex, tool_particle_tex,
										 hashes_tex, cells_start_tex, cells_end_tex, particles->blanked, particles->S,
										 particles->R, p->v_der, p->S_der, p->T_t, p->pos_t, p->vel_t, p->N, fixed_tex,
										 global_dz, p->eps_pl, p->eps_pl_dot, global_time_dt,
										  particles->T, p->vel, particles->fixed, particles->n);

	cleanup_texture_objects();

	check_cuda_error("interactions monaghan\n");
}

void interactions_heat_pse(particle_gpu *particles, int *cells_start, int *cells_end, int num_cell)
{
	if (!m_thermal_workpiece)
		return;

	setup_texture_objects(particles, cells_start, cells_end, num_cell);

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	do_interactions_heat<<<dG, dB>>>(pos_tex, h_tex, T_tex, tool_particle_tex, hashes_tex, cells_start_tex, cells_end_tex, rho_tex,
									 p->T_t, p->N, thermals_workpiece.alpha, thermals_tool.alpha, global_dz);
	cleanup_texture_objects();
}

void interactions_heat_Brookshaw(particle_gpu *particles, int *cells_start, int *cells_end, int num_cell)
{
	if (!m_thermal_workpiece)
		return;

	setup_texture_objects(particles, cells_start, cells_end, num_cell);

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	particle_gpu *p = particles;
	int N = p->N;

	do_interactions_heat_Brookshaw<<<dG, dB>>>(pos_tex, h_tex, T_tex, tool_particle_tex, hashes_tex, cells_start_tex, cells_end_tex, rho_tex,
											   p->T_t, p->N, thermals_workpiece.alpha, thermals_tool.alpha, global_dz);

	cleanup_texture_objects();
}

void interactions_setup_geometry_constants(grid_base *g)
{
	geom_constants geometry_h;
	geometry_h.nx = g->nx();
	geometry_h.ny = g->ny();
	geometry_h.nz = g->nz();
	geometry_h.bbmin_x = g->bbmin_x();
	geometry_h.bbmin_y = g->bbmin_y();
	geometry_h.bbmin_z = g->bbmin_z();
	geometry_h.dx = g->dx();
	cudaMemcpyToSymbol(geometry, &geometry_h, sizeof(geom_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_physical_constants(phys_constants physics_h)
{
	cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
	if (physics_h.mass == 0 || isnan(physics_h.mass))
	{
		printf("WARNING: invalid mass set!\n");
	}
}

void interactions_setup_corrector_constants(corr_constants correctors_h)
{
	cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void interactions_setup_thermal_constants_workpiece(trml_constants trml_h)
{
	thermals_workpiece = trml_h;
	m_thermal_workpiece = trml_h.alpha != 0.;
#if defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE)
	if (m_thermal_workpiece)
	{
		printf("considering thermal diffusion in workpiece\n");
#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_Brookshaw))
		printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
	printf("Diffusitvity workpiece: %e\n", trml_h.alpha);
#endif

	cudaMemcpyToSymbol(trml, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	check_cuda_error("error copying thermal constants.\n");
}

void interactions_setup_thermal_constants_tool(trml_constants trml_h, tool_3d_gpu *tool)
{
	thermals_tool = trml_h;
	m_thermal_tool = trml_h.alpha != 0.;
#if defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_PSE)
	if (m_thermal_tool)
	{
		tool->set_thermal(true);
		printf("considering thermal diffusion from workpiece into tool\n");
#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_Brookshaw))
		printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
	printf("Diffusitvity tool: %e\n", trml_h.alpha);
#endif

	cudaMemcpyToSymbol(trml_tool, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	check_cuda_error("error copying thermal constants.\n");
}

void interactions_setup_thermal_constants_tool(trml_constants trml_h)
{
	thermals_tool = trml_h;
	cudaMemcpyToSymbol(trml_tool, &trml_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	check_cuda_error("error copying thermal constants.\n");
}

void interactions_setup_johnson_cook_constants(joco_constants johnson_cook_h)
{
	cudaMemcpyToSymbol(johnson_cook, &johnson_cook_h, sizeof(joco_constants), 0, cudaMemcpyHostToDevice);
	check_cuda_error("error copying johnson cook constants.\n");
}
