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
// hello
#include "actions_gpu.h"

#include "eigen_solver.cuh"
#include "plasticity.cuh"

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

static bool m_plastic = false;
static bool m_thermal = false;		 // consider thermal conduction in workpiece
static bool m_fric_heat_gen = false; // consider that friction produces heat

__constant__ static phys_constants physics;
__constant__ static corr_constants correctors;
__constant__ static joco_constants johnson_cook;
__constant__ static trml_constants thermals_wp;
__constant__ static trml_constants thermals_tool;

__device__ void updatePosition(float_x x, float_x y, float_x w, float_x t, float_x &x_new, float_x &y_new)
{
	// Convert angular velocity to radians per second
	float_x omega = (w * 2 * M_PI) / 60.0;

	// Calculate the angle of rotation
	float_x theta = omega * t;

	// Compute the new position
	x_new = x * cos(theta) - y * sin(theta);
	y_new = x * sin(theta) + y * cos(theta);
}

__device__ __forceinline__ bool isnaninf(float_x val)
{
	return isnan(val) || isinf(val);
}

__global__ void do_material_eos(const float_x *__restrict__ blanked, const float_x *__restrict__ in_tool,
								const float_x *__restrict__ rho, float_x *__restrict__ p, int N, const float_x *__restrict__ fixed)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	/* if (fixed[pidx] > 1.)
		return; */

	float_x rho0 = physics.rho0;
	float_x c0 = sqrtf(physics.K / rho0);
	float_x rhoi = rho[pidx];
	p[pidx] = c0 * c0 * (rhoi - rho0);
}

__global__ void do_move_tool_particles(particle_gpu particles, float_x vel_z, float_x gwz, float_x dt, float_x probe_vz)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init)
		return;
}

__global__ void do_corrector_artificial_stress(const float_x *__restrict__ blanked, const float_x *__restrict__ in_tool,
											   const float_x *__restrict__ rho, const float_x *__restrict__ p, const mat3x3_t *__restrict__ S,
											   mat3x3_t *__restrict__ R, int N, const float_x *__restrict__ fixed)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	/* if (fixed[pidx] > 1.)
		return; */

	float_x eps = correctors.stresseps;

	if (eps == 0.)
		return;

	float_x rhoi = rho[pidx];
	float_x pi = p[pidx];
	mat3x3_t Si = S[pidx];

	double rhoi21 = 1. / (rhoi * rhoi);

	float_x sxx = Si[0][0] - pi;
	float_x syy = Si[1][1] - pi;
	float_x szz = Si[2][2] - pi;

	float_x sxy = Si[0][1];
	float_x syz = Si[1][2];
	float_x sxz = Si[0][2];

	float3_t eigenvals;
	float3_t e1;
	float3_t e2;
	float3_t e3;

	solve_eigen(sxx, sxy, sxz, syy, syz, szz, eigenvals, e1, e2, e3);

	mat3x3_t Rot(e1.x, e2.x, e3.x,
				 e1.y, e2.y, e3.y,
				 e1.z, e2.z, e3.z);

	mat3x3_t Srot(eigenvals.x, 0., 0.,
				  0., eigenvals.y, 0.,
				  0., 0., eigenvals.z);

	if (Srot[0][0] > 0)
	{
		Srot[0][0] = -eps * Srot[0][0] * rhoi21;
	}
	else
	{
		Srot[0][0] = 0.;
	}

	if (Srot[1][1] > 0)
	{
		Srot[1][1] = -eps * Srot[1][1] * rhoi21;
	}
	else
	{
		Srot[1][1] = 0.;
	}

	if (Srot[2][2] > 0)
	{
		Srot[2][2] = -eps * Srot[2][2] * rhoi21;
	}
	else
	{
		Srot[2][2] = 0.;
	}

	mat3x3_t art_stress = Rot * Srot * glm::transpose(Rot);

	R[pidx] = art_stress;
}

__global__ void do_material_stress_rate_jaumann(const float_x *__restrict__ blanked, const float_x *__restrict__ in_tool,
												const mat3x3_t *__restrict__ v_der, const mat3x3_t *__restrict__ Stress,
												mat3x3_t *S_t, int N, const float_x *__restrict__ fixed)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;

	/* if (fixed[pidx] > 1.)
		return; */

	float_x G = physics.G;

	mat3x3_t vi_der = v_der[pidx];
	mat3x3_t Si = Stress[pidx];

	float_x vx_x = vi_der[0][0];
	float_x vx_y = vi_der[0][1];
	float_x vx_z = vi_der[0][2];

	float_x vy_x = vi_der[1][0];
	float_x vy_y = vi_der[1][1];
	float_x vy_z = vi_der[1][2];

	float_x vz_x = vi_der[2][0];
	float_x vz_y = vi_der[2][1];
	float_x vz_z = vi_der[2][2];

	float_x Sxx = Si[0][0];
	float_x Sxy = Si[0][1];
	float_x Sxz = Si[0][2];

	float_x Syx = Si[1][0];
	float_x Syy = Si[1][1];
	float_x Syz = Si[1][2];

	float_x Szx = Si[2][0];
	float_x Szy = Si[2][1];
	float_x Szz = Si[2][2];

	const mat3x3_t epsdot = mat3x3_t(vx_x, 0.5 * (vx_y + vy_x), 0.5 * (vx_z + vz_x),
									 0.5 * (vx_y + vy_x), vy_y, 0.5 * (vy_z + vz_y),
									 0.5 * (vx_z + vz_x), 0.5 * (vy_z + vz_y), vz_z);

	const mat3x3_t omega = mat3x3_t(0., 0.5 * (vy_x - vx_y), 0.5 * (vz_x - vx_z),
									0.5 * (vx_y - vy_x), 0., 0.5 * (vz_y - vy_z),
									0.5 * (vx_z - vz_x), 0.5 * (vy_z - vz_y), 0.);

	const mat3x3_t S = mat3x3_t(Sxx, Sxy, Sxz,
								Sxy, Syy, Syz,
								Sxz, Syz, Szz);

	const mat3x3_t I = mat3x3_t(1.);

	float_x trace_epsdot = epsdot[0][0] + epsdot[1][1] + epsdot[2][2];

	mat3x3_t Si_t = float_x(2.) * G * (epsdot - float_x(1. / 3.) * trace_epsdot * I) + omega * S + S * glm::transpose(omega); // Belytschko (3.7.9)

	S_t[pidx] = Si_t;
}

__global__ void do_material_fric_heat_gen(const float_x *__restrict__ blanked, const float_x *__restrict__ in_tool,
										  const float4_t *__restrict__ vel, const float3_t *__restrict__ f_fric, const float3_t *__restrict__ n, float_x *__restrict__ T, float3_t vel_tool, float_x dt, int N, const float_x *__restrict__ fixed)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	/* if (fixed[pidx] > 1.)
		return; */

	const float_x eta = thermals_wp.eta;

	// compute F_fric_mag;
	float3_t f_T = f_fric[pidx];
	float_x f_fric_mag = sqrtf(f_T.x * f_T.x + f_T.y * f_T.y + f_T.z * f_T.z);

	if (f_fric_mag == 0.)
	{
		return;
	}

	// compute v_rel
	float3_t normal = n[pidx];
	float4_t v_particle = vel[pidx];
	float3_t v_diff = make_float3_t(v_particle.x - vel_tool.x, v_particle.y - vel_tool.y, v_particle.z - vel_tool.z);

	float_x v_diff_dot_normal = v_diff.x * normal.x + v_diff.y * normal.y + v_diff.z * normal.z;
	float3_t v_relative = make_float3_t(v_diff.x - v_diff_dot_normal, v_diff.y - v_diff_dot_normal, v_diff.z - v_diff_dot_normal);

	float_x v_rel_mag = sqrtf(v_relative.x * v_relative.x + v_relative.y * v_relative.y + v_relative.z * v_relative.z);

	T[pidx] += eta * dt * f_fric_mag * v_rel_mag / (thermals_wp.cp * physics.mass);
}

__global__ void do_contmech_continuity(const float_x *__restrict__ blanked, const float_x *__restrict__ in_tool,
									   const float_x *__restrict__ rho, const mat3x3_t *__restrict__ v_der,
									   float_x *__restrict__ rho_t, int N, const float_x *__restrict__ fixed)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	/* if (fixed[pidx] > 1.)
		return; */

	double rhoi = rho[pidx];
	mat3x3_t vi_der = v_der[pidx];

	float_x vx_x = vi_der[0][0];
	float_x vy_y = vi_der[1][1];
	float_x vz_z = vi_der[2][2];

	rho_t[pidx] = -rhoi * (vx_x + vy_y + vz_z);
}

__global__ void do_contmech_momentum(const float_x *__restrict__ blanked, const float_x *__restrict__ in_tool,
									 const mat3x3_t *__restrict__ S_der, const float3_t *__restrict__ fc, const float3_t *__restrict__ ft,
									 float3_t *__restrict__ vel_t, int N, const float_x *__restrict__ fixed)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	/* if (fixed[pidx] > 1.)
		return; */

	float_x mass = physics.mass;

	mat3x3_t Si_der = S_der[pidx];
	float3_t fci = fc[pidx];
	float3_t fti = ft[pidx];
	float3_t veli_t = vel_t[pidx];

	float_x Sxx_x = Si_der[0][0];
	float_x Sxy_y = Si_der[0][1];
	float_x Sxz_z = Si_der[0][2];

	float_x Syx_x = Si_der[1][0];
	float_x Syy_y = Si_der[1][1];
	float_x Syz_z = Si_der[1][2];

	float_x Szx_x = Si_der[2][0];
	float_x Szy_y = Si_der[2][1];
	float_x Szz_z = Si_der[2][2];

	float_x fcx = fci.x;
	float_x fcy = fci.y;
	float_x fcz = fci.z;

	float_x ftx = fti.x;
	float_x fty = fti.y;
	float_x ftz = fti.z;

	// redundant mult and div by rho elimnated in der stress
	veli_t.x += Sxx_x + Sxy_y + Sxz_z + fcx / mass + ftx / mass;
	veli_t.y += Syx_x + Syy_y + Syz_z + fcy / mass + fty / mass;
	veli_t.z += Szx_x + Szy_y + Szz_z + fcz / mass + ftz / mass;

	vel_t[pidx] = veli_t;
}

__global__ void do_contmech_advection(const float_x *__restrict__ blanked, const float_x *__restrict__ in_tool,
									  const float4_t *__restrict__ vel,
									  float3_t *__restrict__ pos_t, int N, const float_x *__restrict__ fixed)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= N)
		return;
	/* if (fixed[pidx] > 1.)
		return; */

	float4_t veli = vel[pidx];
	float3_t posi_t = pos_t[pidx];

	float3_t posi_t_new;
	posi_t_new.x = posi_t.x + veli.x;
	posi_t_new.y = posi_t.y + veli.y;
	posi_t_new.z = posi_t.z + veli.z;

	pos_t[pidx] = posi_t_new;
}

__global__ void do_invalidate_rate(particle_gpu particles)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N)
		return;

	bool invalid = false;

	invalid = invalid || isnaninf(particles.pos_t[pidx].x);
	invalid = invalid || isnaninf(particles.pos_t[pidx].y);
	invalid = invalid || isnaninf(particles.pos_t[pidx].z);

	invalid = invalid || isnaninf(particles.vel_t[pidx].x);
	invalid = invalid || isnaninf(particles.vel_t[pidx].y);
	invalid = invalid || isnaninf(particles.vel_t[pidx].z);

	invalid = invalid || isnaninf(particles.S_t[pidx][0][0]);
	invalid = invalid || isnaninf(particles.S_t[pidx][1][1]);
	invalid = invalid || isnaninf(particles.S_t[pidx][2][2]);

	invalid = invalid || isnaninf(particles.S_t[pidx][0][1]);
	invalid = invalid || isnaninf(particles.S_t[pidx][1][2]);
	invalid = invalid || isnaninf(particles.S_t[pidx][2][0]);

	invalid = invalid || isnaninf(particles.rho_t[pidx]);
	invalid = invalid || isnaninf(particles.T_t[pidx]);

	if (invalid)
	{
		particles.blanked[pidx] = 1.;
		printf("invalidated particle %d due to nan!\n", pidx);
	}
}

__global__ void do_check_valid_full(particle_gpu particles)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N)
		return;

	if (particles.blanked[pidx] == 1.)
	{
		return;
	}

	bool invalid = false;

	invalid = invalid || isnaninf(particles.pos[pidx].x);
	invalid = invalid || isnaninf(particles.pos[pidx].y);
	invalid = invalid || isnaninf(particles.pos[pidx].z);

	invalid = invalid || isnaninf(particles.vel[pidx].x);
	invalid = invalid || isnaninf(particles.vel[pidx].y);
	invalid = invalid || isnaninf(particles.vel[pidx].z);

	invalid = invalid || isnaninf(particles.S[pidx][0][0]);
	invalid = invalid || isnaninf(particles.S[pidx][1][1]);
	invalid = invalid || isnaninf(particles.S[pidx][2][2]);

	invalid = invalid || isnaninf(particles.S[pidx][0][2]);
	invalid = invalid || isnaninf(particles.S[pidx][0][1]);
	invalid = invalid || isnaninf(particles.S[pidx][1][2]);

	invalid = invalid || isnaninf(particles.rho[pidx]);
	invalid = invalid || isnaninf(particles.T[pidx]);

	invalid = invalid || isnaninf(particles.eps_pl[pidx]);
	invalid = invalid || isnaninf(particles.eps_pl_dot[pidx]);
	invalid = invalid || isnaninf(particles.p[pidx]);

	if (invalid)
	{
		printf("found particle with nan values that is not blanked!\n");
	}
}

__global__ void do_plasticity_johnson_cook(particle_gpu particles, float_x dt, float_x global_Vsf)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N)
		return;
	/* if (particles.fixed[pidx] > 1.)
		return; */

	/* if(particles.tool_particle[pidx] == 1. && particles.fc[pidx].z !=0)
		printf("contact force %f\n", particles.fc[pidx].z); */

	float_x mu = physics.G;

	mat3x3_t S = particles.S[pidx];
	float_x Strialxx = S[0][0];
	float_x Strialyy = S[1][1];
	float_x Strialzz = S[2][2];
	float_x Strialxy = S[0][1];
	float_x Strialyz = S[1][2];
	float_x Strialzx = S[2][0];

	float_x norm_Strial = sqrt_t(Strialxx * Strialxx + Strialyy * Strialyy + Strialzz * Strialzz + 2 * (Strialxy * Strialxy + Strialyz * Strialyz + Strialzx * Strialzx));

	float_x eps_pl = particles.eps_pl[pidx];
	float_x eps_pl_dot = particles.eps_pl_dot[pidx];
	float_x T = particles.T[pidx];

	if (johnson_cook.clamp_temp)
	{
		T = (T > johnson_cook.Tmelt) ? johnson_cook.Tmelt - 1 : T;
	}
	else
	{
		if (T > johnson_cook.Tmelt)
		{
			printf("Particle melted!\n");
		}
	}

	float_x svm = sqrt_t(3.0 / 2.0) * norm_Strial;

	float_x sigma_Y = sigma_yield(johnson_cook, eps_pl, eps_pl_dot, T);

	// elastic case
	if (svm < sigma_Y)
	{
		particles.eps_pl_dot[pidx] = 0.;
		return;
	}

	bool failed = false;
	float_x delta_lambda = solve_secant(johnson_cook, fmax(eps_pl_dot * dt * sqrt(2. / 3.), 1e-8), 1e-6,
										norm_Strial, eps_pl, T, dt, physics.G, failed);

	if (failed)
	{
		printf("%d: %f %f %f: eps_pl %f, eps_pl_dot %f, T %f\n", pidx, particles.pos[pidx].x, particles.pos[pidx].y, particles.pos[pidx].z,
			   particles.eps_pl[pidx], particles.eps_pl_dot[pidx], particles.T[pidx]);
	}

	float_x eps_pl_new = eps_pl + sqrtf(2.0 / 3.0) * fmaxf(delta_lambda, 0.);
	float_x eps_pl_dot_new = (sqrtf(2.0 / 3.0) * fmaxf(delta_lambda, 0.) / dt) / global_Vsf;

	particles.eps_pl[pidx] = eps_pl_new;
	particles.eps_pl_dot[pidx] = eps_pl_dot_new;

	mat3x3_t S_new;
	S_new[0][0] = Strialxx - Strialxx / norm_Strial * delta_lambda * 2. * mu;
	S_new[1][1] = Strialyy - Strialyy / norm_Strial * delta_lambda * 2. * mu;
	S_new[2][2] = Strialzz - Strialzz / norm_Strial * delta_lambda * 2. * mu;

	S_new[0][1] = Strialxy - Strialxy / norm_Strial * delta_lambda * 2. * mu;
	S_new[1][0] = Strialxy - Strialxy / norm_Strial * delta_lambda * 2. * mu;

	S_new[1][2] = Strialyz - Strialyz / norm_Strial * delta_lambda * 2. * mu;
	S_new[2][1] = Strialyz - Strialyz / norm_Strial * delta_lambda * 2. * mu;

	S_new[2][0] = Strialzx - Strialzx / norm_Strial * delta_lambda * 2. * mu;
	S_new[0][2] = Strialzx - Strialzx / norm_Strial * delta_lambda * 2. * mu;

	particles.S[pidx] = S_new;

	// plastic work to heatinteractions_calculate_force_die_using_kirk_method
	if (thermals_wp.tq != 0.)
	{
		float_x delta_eps_pl = eps_pl_new - eps_pl;
		float_x sigma_Y = sigma_yield(johnson_cook, eps_pl_new, eps_pl_dot_new, T);
		float_x rho = particles.rho[pidx];
		particles.T[pidx] += thermals_wp.tq / (thermals_wp.cp * rho) * delta_eps_pl * sigma_Y;
	}
}

__global__ void do_boundary_conditions_thermal(particle_gpu particles, float_x dz, float_x dt, float_x Vsf)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N)
		return;

	int fixed = (int)particles.fixed[pidx];

	// Only process relevant fixed values
	if (fixed == -1)
	{
		float_x T = particles.T[pidx];
		float_x mass = particles.rho[pidx] * dz * dz * dz;

		float_x T_ref = thermals_wp.T_init;
		float_x As = dz * dz;
		float_x cp = thermals_tool.cp;


		float_x h_conv = (T > T_ref) ? 1000. * Vsf * 1.e3 : 0.;
		float_x q_dot = h_conv * (T_ref - T) * As;

		particles.T[pidx] += (q_dot * dt) / (cp * mass);
	}
	
	if (fixed == 7 || fixed == 2)
	{
		float_x T = particles.T[pidx];
		float_x mass = particles.rho[pidx] * dz * dz * dz;

		float_x T_ref = thermals_wp.T_init;
		float_x As = dz * dz;
		float_x cp = thermals_tool.cp;


		float_x h_conv = (T > T_ref) ? 5. * Vsf * 1.e3 : 0.;
		float_x q_dot = h_conv * (T_ref - T) * As;

		particles.T[pidx] += (q_dot * dt) / (cp * mass);
	}
}

__global__ void do_boundary_conditions(particle_gpu particles, float_x substrate_velocity, float_x dt, float_x gwz, float_x probe_vz)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N)
		return;
	if (particles.blanked[pidx] == 1.)
		return;

	if (particles.fixed[pidx] == 2)
	{
		particles.vel[pidx].x = substrate_velocity;
		particles.vel[pidx].y = 0;
		particles.vel[pidx].z = 0;

		// particles.pos[pidx].x += substrate_velocity * dt;

		particles.fc[pidx].x = 0.;
		particles.fc[pidx].y = 0.;
		particles.fc[pidx].z = 0.;

		particles.pos_t[pidx].x = 0.;
		particles.pos_t[pidx].y = 0.;
		particles.pos_t[pidx].z = 0.;

		particles.vel_t[pidx].x = 0.;
		particles.vel_t[pidx].y = 0.;
		particles.vel_t[pidx].z = 0.;
		return;
	}

	if (particles.fixed[pidx] == -1)
	{
		float_x px = particles.pos[pidx].x;
		float_x py = particles.pos[pidx].y;

		/* updatePosition(particles.pos[pidx].x, particles.pos[pidx].y, gwz, dt, px, py);
		particles.pos[pidx].x = px;
		particles.pos[pidx].y = py;
		particles.pos[pidx].z += probe_vz * dt; */

		glm::vec3 r(px, py, 0.0);
		glm::vec3 w(0, 0, gwz);
		glm::vec3 v = glm::cross(w, r);

		particles.vel[pidx].x = v.x;
		particles.vel[pidx].y = v.y;
		particles.vel[pidx].z = probe_vz;

		particles.fc[pidx].x = 0.;
		particles.fc[pidx].y = 0.;
		particles.fc[pidx].z = 0.;

		particles.pos_t[pidx].x = 0.;
		particles.pos_t[pidx].y = 0.;
		particles.pos_t[pidx].z = 0.;

		particles.vel_t[pidx].x = 0.;
		particles.vel_t[pidx].y = 0.;
		particles.vel_t[pidx].z = 0.;
		return;
	}
}

__global__ void do_blanking(particle_gpu particles, float_x vel_max_squared, vec3_t bbmin, vec3_t bbmax)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N_init)
		return;

	float4_t pos = particles.pos[pidx];
	float4_t vel = particles.vel[pidx];

	bool too_fast = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z >= vel_max_squared;

	bool in_x = pos.x >= bbmin.x && pos.x <= bbmax.x;
	bool in_y = pos.y >= bbmin.y && pos.y <= bbmax.y;
	bool in_z = pos.z >= bbmin.z && pos.z <= bbmax.z;

	if (too_fast || !in_x || !in_y || !in_z)
	{
		particles.blanked[pidx] = 1.;
	}
	else
	{

		bool invalid = false;

		invalid = invalid || isnaninf(particles.pos_t[pidx].x);
		invalid = invalid || isnaninf(particles.pos_t[pidx].y);
		invalid = invalid || isnaninf(particles.pos_t[pidx].z);

		invalid = invalid || isnaninf(particles.vel_t[pidx].x);
		invalid = invalid || isnaninf(particles.vel_t[pidx].y);
		invalid = invalid || isnaninf(particles.vel_t[pidx].z);

		invalid = invalid || isnaninf(particles.S_t[pidx][0][0]);
		invalid = invalid || isnaninf(particles.S_t[pidx][1][1]);
		invalid = invalid || isnaninf(particles.S_t[pidx][2][2]);

		invalid = invalid || isnaninf(particles.S_t[pidx][0][1]);
		invalid = invalid || isnaninf(particles.S_t[pidx][1][2]);
		invalid = invalid || isnaninf(particles.S_t[pidx][2][0]);

		invalid = invalid || isnaninf(particles.rho_t[pidx]);
		invalid = invalid || isnaninf(particles.T_t[pidx]);

		if (!invalid)
		{ // never, ever, unblank a particle with nan or inf values
			particles.blanked[pidx] = 0.;
		}
	}
}

__device__ void kirk_contact_force(vec3_t &fN_each, float_x pd, float3_t vij, float3_t nav, float_x dt, float_x p_temp)
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
__device__ double sigma_yield_interaction(joco_constants jc, double eps_pl, double eps_pl_dot, double t)
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

__device__ void calculate_contact_force(bool &is_sticking, vec3_t &fN, vec3_t &fT, vec3_t &vr, vec3_t &fricold, float_x gN, vec3_t normal, vec3_t vs, float_x dt, float_x p_temp, float_x contact_alpha, float_x slave_mass, float_x friction_mu, float_x ffl, float_x cp, particle_gpu &particles, unsigned int pidx, float_x &T)
{
	vec3_t kdeltae = contact_alpha * slave_mass * vr / dt;
	float_x fy = friction_mu * glm::length(fN);
	vec3_t fstar = fricold - kdeltae;
	float_x fstar_mag = glm::length(fstar);
	is_sticking = false;

	if (fstar_mag != 0)
	{
		if (fy > ffl)
		{
			fy = ffl;
			is_sticking = true;
		}

		vec3_t fT_t = fy * (fstar / fstar_mag);
		fT += fT_t;
	}
}

__global__ void do_contact_froce(particle_gpu particles, float_x dt,
								 float_x shoulder_surface, float_x shoulder_velocity,
								 float_x shoulder_radius, float_x dz, float_x wz, float_x probe_radius,
								 float_x ring_raduis, float_x top_surface, float_x probe_surface,
								 float_x probe_velocity, float_x substrate_velocity)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N || particles.blanked[pidx] == 1.)
		return;
	if (particles.fixed[pidx] > 1.)
		return;

	if (particles.tool_particle[pidx] == 1.)
		return;

	/* if (particles.fixed[pidx] != 0.)
		return; */

	float4_t pi = particles.pos[pidx];

	// Precompute values
	float_x p_temp = particles.T[pidx];
	float_x eps_pl = particles.eps_pl[pidx];
	float_x eps_pl_dot = particles.eps_pl_dot[pidx];
	float_x sigma_Y = sigma_yield_interaction(johnson_cook, eps_pl, eps_pl_dot, p_temp);
	float_x ffl = (sigma_Y / sqrtf(3.0)) * dz * dz;

	vec3_t vs(particles.vel[pidx].x, particles.vel[pidx].y, particles.vel[pidx].z);
	vec3_t fricold(particles.ft[pidx].x, particles.ft[pidx].y, particles.ft[pidx].z);

	vec3_t fN = {0.0, 0.0, 0.0};
	vec3_t fT = {0.0, 0.0, 0.0};

	const float_x contact_alpha = 0.5;
	const float_x friction_mu = 0.3;

	// under the fixed ring
	if (pi.z < top_surface)
	{
		float3_t normal = particles.n[pidx];
		float_x gN = top_surface - pi.z;
		vec3_t vm;
		vm.x = substrate_velocity;
		vm.y = 0.0;
		vm.z = 0.0;
		// Compute relative velocity
		float4_t v_particle = particles.vel[pidx];
		float3_t v_diff = make_float3_t(v_particle.x - vm.x, v_particle.y - vm.y, v_particle.z - vm.z);
		float_x v_diff_dot_normal = v_diff.x * normal.x + v_diff.y * normal.y + v_diff.z * normal.z;
		float3_t v_relative = make_float3_t(v_diff.x - v_diff_dot_normal, v_diff.y - v_diff_dot_normal, v_diff.z - v_diff_dot_normal);
		float_x v_rel_mag = sqrtf(v_relative.x * v_relative.x + v_relative.y * v_relative.y + v_relative.z * v_relative.z);
		// Compute normal contact force
		kirk_contact_force(fN, gN, v_diff, normal, dt, p_temp);
		// Compute tangential contact force
		vec3_t v_relative_vec = {v_relative.x, v_relative.y, v_relative.z};
		vec3_t kdeltae = contact_alpha * physics.mass * v_relative_vec / dt;
		vec3_t fstar = fricold - kdeltae;
		float_x fstar_mag = glm::length(fstar);
		float_x fy = friction_mu * glm::length(fN);

		if (fstar_mag > 0.0f)
		{
			if (fy > ffl)
			{
				fy = ffl;

				if (particles.T[pidx] >= 0.85 * johnson_cook.Tmelt /* || gN > 0.5*dz */)
				{

					particles.vel[pidx] = make_float4_t(vm.x, vm.y, vm.z, 1);
					particles.fixed[pidx] = 1.;

					particles.fc[pidx].x = 0.;
					particles.fc[pidx].y = 0.;
					particles.fc[pidx].z = 0.;

					particles.pos_t[pidx].x = 0.;
					particles.pos_t[pidx].y = 0.;
					particles.pos_t[pidx].z = 0.;

					particles.vel_t[pidx].x = 0.;
					particles.vel_t[pidx].y = 0.;
					particles.vel_t[pidx].z = 0.;
				}
			}
			particles.T[pidx] += thermals_wp.eta * dt * fy * v_rel_mag / (thermals_wp.cp * physics.mass);
			vec3_t fT_t = fy * (fstar / fstar_mag);
			fT += fT_t;
		}
	}

	// Update particle forces
	particles.fc[pidx] = make_float3_t(fN.x, fN.y, fN.z);
	particles.ft[pidx] = make_float3_t(fT.x, fT.y, fT.z);
}

__global__ void do_merge(particle_gpu particles, float_x dz, float_x substrate_velocity, float_x top_surface)
{
	unsigned int pidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (pidx >= particles.N)
		return;
	if (particles.tool_particle[pidx] != .0)
		return;

	float_x eps_pl = particles.eps_pl[pidx];
	float_x eps_pl_dot = particles.eps_pl_dot[pidx];
	float_x T = particles.T[pidx];
	float_x A_shear = dz * dz;

	float_x ft_x = particles.ft[pidx].x;
	float_x ft_y = particles.ft[pidx].y;
	float_x ft_z = particles.ft[pidx].z;

	float_x ffl = (sigma_yield(johnson_cook, eps_pl, eps_pl_dot, T) * A_shear) / sqrt_t(3.0);
	float_x shear_force = sqrt_t(ft_x * ft_x + ft_y * ft_y + ft_z * ft_z);

	if ( /*(T > 0.85 * johnson_cook.Tmelt) &&*/ (shear_force > ffl)  && (abs(particles.pos[pidx].z) < top_surface + 3.5) )
	{
		particles.vel[pidx] = make_float4_t(substrate_velocity, 0, 0, 1);
		particles.tool_particle[pidx] = 1.;
		particles.fixed[pidx] = 1.;

		particles.fc[pidx].x = 0.;
		particles.fc[pidx].y = 0.;
		particles.fc[pidx].z = 0.;

		particles.pos_t[pidx].x = 0.;
		particles.pos_t[pidx].y = 0.;
		particles.pos_t[pidx].z = 0.;

		particles.vel_t[pidx].x = 0.;
		particles.vel_t[pidx].y = 0.;
		particles.vel_t[pidx].z = 0.;
	}
}

//---------------------------------------------------------------------

// float2 + struct
struct add_float4
{
	__device__ float4_t operator()(const float4_t &a, const float4_t &b) const
	{
		float4_t r;
		r.x = a.x + b.x;
		r.y = a.y + b.y;
		r.z = a.z + b.z;
		return r;
	}
};

void perform_merge_conditions_thermal(particle_gpu *particles)
{
	if (!m_thermal)
		return;
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_merge<<<dG, dB>>>(*particles, global_dz, global_substrate_velocity, global_top_surface);
	check_cuda_error("After perform_merge_conditions_thermal\n");
}

void contact_force(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_contact_froce<<<dG, dB>>>(*particles, global_time_dt, global_shoulder_contact_surface, global_substrate_velocity,
								 global_shoulder_raduis, global_dz, global_wz, global_probe_raduis, global_ring_raduis, global_top_surface,
								 global_probe_contact_surface, global_probe_velocity, global_substrate_velocity);
	check_cuda_error("After material_eos\n");
}

void material_eos(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_material_eos<<<dG, dB>>>(particles->blanked, particles->tool_particle, particles->rho, particles->p, particles->N, particles->fixed);
	check_cuda_error("After material_eos\n");
}

void corrector_artificial_stress(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_corrector_artificial_stress<<<dG, dB>>>(particles->blanked, particles->tool_particle, particles->rho, particles->p, particles->S, particles->R, particles->N, particles->fixed);
	check_cuda_error("After Corrector Artifical Stress\n");
}

void material_stress_rate_jaumann(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_material_stress_rate_jaumann<<<dG, dB>>>(particles->blanked, particles->tool_particle, particles->v_der, particles->S, particles->S_t, particles->N, particles->fixed);
	check_cuda_error("After material_stress_rate_jaumann\n");
}

void contmech_continuity(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_contmech_continuity<<<dG, dB>>>(particles->blanked, particles->tool_particle, particles->rho, particles->v_der, particles->rho_t, particles->N, particles->fixed);
	check_cuda_error("After contmech_continuity\n");
}

void contmech_momentum(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_contmech_momentum<<<dG, dB>>>(particles->blanked, particles->tool_particle, particles->S_der, particles->fc, particles->ft, particles->vel_t, particles->N, particles->fixed);
	check_cuda_error("After contmech_momentum\n");
}

void contmech_advection(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_contmech_advection<<<dG, dB>>>(particles->blanked, particles->tool_particle, particles->vel, particles->pos_t, particles->N, particles->fixed);
	check_cuda_error("After contmech_advection\n");
}

void plasticity_johnson_cook(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_plasticity_johnson_cook<<<dG, dB>>>(*particles, global_time_dt, global_Vsf);
	check_cuda_error("After johnson_cook\n");
}

void perform_boundary_conditions_thermal(particle_gpu *particles)
{
	if (!m_thermal)
		return;

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_boundary_conditions_thermal<<<dG, dB>>>(*particles, global_dz, global_time_dt, global_Vsf);
	check_cuda_error("After boundary_conditions_thermal\n");
}

void perform_boundary_conditions(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	/* float4_t *h_pos = 0;
	float4_t *h_vel = 0;
	unsigned int n = particles->N;
	h_pos = new float4_t[n];
	h_vel = new float4_t[n];
	cudaMemcpy(h_pos, particles->pos, sizeof(float4_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vel, particles->vel, sizeof(float4_t) * n, cudaMemcpyDeviceToHost);

	float_x mean_z = 0;
	int count = 0;
	for (size_t i = 0; i < n; i++)
	{
		if (h_vel[i].w == 1)
		{
			mean_z += h_pos[i].z;
			count++;
		}
	}

	if (count > 0)
	{
		mean_z /= count;
		global_top_surface = mean_z + 1. * global_dz;
	}

	delete[] h_pos;
	delete[] h_vel; */

	do_boundary_conditions<<<dG, dB>>>(*particles, global_substrate_velocity, global_time_dt, global_wz, global_probe_velocity);
	cudaThreadSynchronize();
	;
	check_cuda_error("After boundary_conditions\n");
}

void actions_move_tool_particles(particle_gpu *particles, tool_3d_gpu *tool_old)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N_init + block_size - 1) / block_size);
	dim3 dB(block_size);

	// if (global_time_current > 0.21 && global_time_current <= 0.24)
	//	global_substrate_velocity = 0.; // stop it for a while

	// if (global_time_current > 0.24)
	//	global_substrate_velocity = -global_probe_velocity/1.25; // Reverse the direction of velocity

	/* if (global_time_current > global_time_final/2. && global_probe_velocity > 0)
	{

		global_probe_velocity = -1. * global_probe_velocity; // Reverse the direction of velocity
		global_substrate_velocity = -1. * global_substrate_velocity;
	}

	global_shoulder_contact_surface += global_substrate_velocity * global_time_dt;
	global_probe_contact_surface += global_probe_velocity * global_time_dt; */

	do_move_tool_particles<<<dG, dB>>>(*particles, global_substrate_velocity, global_wz, global_time_dt, global_probe_velocity);
	cudaThreadSynchronize();
}

void perform_blanking(particle_gpu *particles, blanking *global_blanking)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N_init + block_size - 1) / block_size);
	dim3 dB(block_size);

	vec3_t bbmin, bbmax;
	global_blanking->get_bb(bbmin, bbmax);
	do_blanking<<<dG, dB>>>(*particles, global_blanking->get_max_vel_squared(), bbmin, bbmax);
	check_cuda_error("After blanking\n");
}

void perform_blanking_dbg(particle_gpu *particles, blanking *global_blanking)
{

	static float_x *h_blanking = 0;

	if (h_blanking == 0)
	{
		h_blanking = new float_x[particles->N_init];
	}

	for (int i = 0; i < particles->N_init; i++)
	{
		float_x rand_num = rand() / ((float_x)RAND_MAX);
		h_blanking[i] = (rand_num > 0.5) ? 1 : 0;
	}

	cudaMemcpy(particles->blanked, h_blanking, sizeof(float_x) * particles->N_init, cudaMemcpyHostToDevice);
	check_cuda_error("after blanking dbg!\n");
}

void actions_setup_physical_constants(phys_constants physics_h)
{
	if (physics_h.mass == 0 || isnan(physics_h.mass))
	{
		printf("WARNING: invalid mass set!\n");
	}

	cudaMemcpyToSymbol(physics, &physics_h, sizeof(phys_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_corrector_constants(corr_constants correctors_h)
{

	if (correctors_h.stresseps > 0.)
	{
		printf("using artificial stresses with eps %f\n", correctors_h.stresseps);
	}

	cudaMemcpyToSymbol(correctors, &correctors_h, sizeof(corr_constants), 0, cudaMemcpyHostToDevice);
}

void actions_setup_johnson_cook_constants(joco_constants johnson_cook_h)
{
	cudaMemcpyToSymbol(johnson_cook, &johnson_cook_h, sizeof(joco_constants), 0, cudaMemcpyHostToDevice);
	m_plastic = true;
}

void actions_setup_thermal_constants_wp(trml_constants thermal_h)
{
	cudaMemcpyToSymbol(thermals_wp, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);
	if (thermal_h.tq != 0.)
	{
		printf("considering generation of heat due to plastic work\n");
	}

	if (thermal_h.eta != 0.)
	{
		printf("considering that friction generates heat\n");
		m_fric_heat_gen = true;
	}

	if (thermal_h.alpha != 0.)
	{
		m_thermal = true;
#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_Brookshaw))
		printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
}

void actions_setup_thermal_constants_tool(trml_constants thermal_h)
{
	cudaMemcpyToSymbol(thermals_tool, &thermal_h, sizeof(trml_constants), 0, cudaMemcpyHostToDevice);

	if (thermal_h.alpha != 0.)
	{
		m_thermal = true;

#if !(defined(Thermal_Conduction_Brookshaw) || defined(Thermal_Conduction_Brookshaw))
		printf("warning! heat conduction constants set but no heat conduction algorithm active!");
#endif
	}
}

void material_fric_heat_gen(particle_gpu *particles, vec3_t vel)
{
	if (!m_fric_heat_gen)
	{
		return;
	}

	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);
	do_material_fric_heat_gen<<<dG, dB>>>(particles->blanked, particles->tool_particle, particles->vel, particles->ft, particles->n,
										  particles->T, make_float3_t(vel.x, vel.y, vel.z), global_time_dt, particles->N, particles->fixed);
	check_cuda_error("After material_fric_heat_gen\n");
}

void debug_check_valid(particle_gpu *particles)
{
	thrust::device_ptr<float4_t> t_pos(particles->pos);
	float4_t ini;
	ini.x = 0.;
	ini.y = 0.;
	ini.z = 0.;
	ini = thrust::reduce(t_pos, t_pos + particles->N, ini, add_float4());

	if (isnan(ini.x) || isnan(ini.y) || isnan(ini.z))
	{
		printf("nan found!\n");
		exit(-1);
	}
}

void debug_check_valid_full(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	do_check_valid_full<<<dG, dB>>>(*particles);
}

void debug_invalidate(particle_gpu *particles)
{
	const unsigned int block_size = BLOCK_SIZE;
	dim3 dG((particles->N + block_size - 1) / block_size);
	dim3 dB(block_size);

	do_invalidate_rate<<<dG, dB>>>(*particles);
	cudaThreadSynchronize();
}
