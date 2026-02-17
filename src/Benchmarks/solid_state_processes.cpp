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

#include "solid_state_processes.h"

struct Point
{
	float_x x;
	float_x y;
	float_x z;
};

void generate_circular_arrangement(float_x n, float_x r, float_x z, std::vector<Point> &points)
{

	if (r == 0)
	{
		Point point;
		point.x = r;
		point.y = r;
		point.z = z;
		points.push_back(point);
	}

	for (int i = 0; i < n; i++)
	{
		double angle = 2 * M_PI * i / n;
		Point point;
		point.x = r * cos(angle);
		point.y = r * sin(angle);
		point.z = z;
		points.push_back(point);
	}
}

float_x distance(Point a, Point b)
{

	float deff_x = a.x - b.x;
	float deff_y = a.y - b.y;
	return sqrt(deff_x * deff_x + deff_y * deff_y);
}

void generate_circular_points(float_x diameter, float_x dz, float_x zz, std::vector<Point> &points)
{
	float_x r = 0.0;
	while (r < diameter / 2.0)
	{
		int n = static_cast<int>((2 * M_PI * r) / dz);
		generate_circular_arrangement(n, r, zz, points);
		r += dz;
	}
}

void generate_circular_points_hollow(float_x outer_diameter, float_x dz, float_x zz, std::vector<Point> &points)
{
	float_x outer_radius = outer_diameter / 2.0;

	for (float_x x = -outer_radius; x <= outer_radius; x += dz)
	{
		for (float_x y = -outer_radius; y <= outer_radius; y += dz)
		{
			float_x r = sqrt(x * x + y * y);
			if (r <= outer_radius)
			{
				points.push_back(Point{x, y, zz});
			}
		}
	}
}

void generate_grid_points(int nx, int ny, float_x dz, float_x zz, std::vector<Point> &points)
{
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j)
		{
			float_x x = -nx / 2.0 * dz + i * dz;
			float_x y = -ny / 2.0 * dz + j * dz;
			points.push_back(Point{x, y, zz});
		}
	}
}

float_x find_lowest_point_z(const std::vector<Point> &points, float_x back_plate_hight)
{
	if (points.empty())
	{
		throw std::runtime_error("The points vector is empty.");
	}

	printf("find lowest point z\n");
	printf("back_plate_hight: %f\n", back_plate_hight);

	float_x lowest_point = std::numeric_limits<float_x>::max();
	for (const auto &point : points)
	{
		if (point.z < back_plate_hight)
		{
			continue;
		}
		if (point.z < lowest_point)
		{
			lowest_point = point.z;
		}
	}

	return lowest_point;
}

float_x find_lowest_point_z(const std::vector<Point> &points)
{
	if (points.empty())
	{
		throw std::runtime_error("The points vector is empty.");
	}

	printf("find lowest point z\n");

	float_x lowest_point = std::numeric_limits<float_x>::max();
	for (const auto &point : points)
	{

		if (point.z < lowest_point)
		{
			lowest_point = point.z;
		}
	}

	return lowest_point;
}

void find_min_max_xy(const std::vector<Point> &points, float_x &min_x, float_x &max_x, float_x &min_y, float_x &max_y,
					 float_x &min_z, float_x &max_z)
{
	if (points.empty())
	{
		throw std::runtime_error("The points vector is empty.");
	}

	min_x = std::numeric_limits<float_x>::max();
	max_x = std::numeric_limits<float_x>::lowest();
	min_y = std::numeric_limits<float_x>::max();
	max_y = std::numeric_limits<float_x>::lowest();
	min_z = std::numeric_limits<float_x>::max();
	max_z = std::numeric_limits<float_x>::lowest();

	for (const auto &point : points)
	{
		if (point.x < min_x)
			min_x = point.x;
		if (point.x > max_x)
			max_x = point.x;
		if (point.y < min_y)
			min_y = point.y;
		if (point.y > max_y)
			max_y = point.y;
		if (point.z < min_z)
			min_z = point.z;
		if (point.z > max_z)
			max_z = point.z;
	}
}

particle_gpu *setup_FS(int nbox, grid_base **grid)
{
	phys_constants phys = make_phys_constants();
	corr_constants corr = make_corr_constants();
	trml_constants trml_wp = make_trml_constants();
	trml_constants trml_tool = make_trml_constants();
	joco_constants joco = make_joco_constants();

	// Constants
	constexpr float_x ms = 1.0;
	global_Vsf = 10.;
	constexpr float_x dz = 1.0;
	global_dz = dz;
	constexpr float_x hdx = 1.7;

	// dimensions of the workpiece
	constexpr float_x wp_width = 50.0;
	constexpr float_x wp_length = 100.0;
	constexpr float_x wp_thickness = 8.0 + 2. * dz;
	constexpr float_x probe_diameter = 20.0;
	global_probe_raduis = probe_diameter / 2.0;
	constexpr float_x probe_hight = 40.0;
	float_t shift = wp_length / 4.0;


	int nx = static_cast<int>(wp_length / dz) + 1;
	int ny = static_cast<int>(wp_width / dz) + 1;

	// BC
	global_probe_velocity = -1.8 * global_Vsf;
	global_substrate_velocity = 6. * global_Vsf;

	global_wz = 900. * 0.104719755 * global_Vsf;
	glm::vec3 w(0.0, 0.0, global_wz);

	// physical constants
	phys.E = 70.e9;
	phys.nu = 0.3;
	phys.rho0 = 2700.0 * 1.0e-6 * ms;
	phys.G = phys.E / (2. * (1. + phys.nu));
	phys.K = 2.0 * phys.G * (1 + phys.nu) / (3 * (1 - 2 * phys.nu));
	phys.mass = dz * dz * dz * phys.rho0;
	

	// Johnson Cook Constants rod & substrate Al5083H116 // DOI: 10.1177/0021998315622982
	joco.A = 167.e6; 
	joco.B = 0.;		//	596e6
	joco.C = 0.001;		
	joco.m = 0.859;		
	joco.n = 0.;		
	joco.Tref = 20.;	
	joco.Tmelt = 620.;	
	joco.eps_dot_ref = 1;
	joco.clamp_temp = 1.;

	// Thermal Constants
	trml_wp.cp = (910. * 1.0e6) / ms;					  // Heat Capacity
	trml_wp.tq = 0.9;									  // Taylor-Quinney Coefficient
	trml_wp.k = 117. * 1.0e6 * global_Vsf;				  // Thermal Conduction
	trml_wp.alpha = trml_wp.k / (phys.rho0 * trml_wp.cp); // Thermal diffusivity
	trml_wp.eta = 0.9;
	trml_wp.T_init = joco.Tref; // Initial Temperature

	// Corrector Constants
	corr.alpha = 1.;
	corr.beta = 1.;
	corr.eta = 0.1;
	corr.xspheps = 0.01;
	corr.stresseps = 0.3;
	float_x h1 = 1. / (hdx * dz);
	float_x q = dz * h1;
	float_x fac = (M_1_PI)*h1 * h1 * h1;
	corr.wdeltap = fac * (1 - 1.5 * q * q * (1 - 0.5 * q));

	std::vector<Point> points;
	float_x zz = 0;

	while (zz < wp_thickness + probe_hight)
	{
		if (zz < wp_thickness)
		{
			generate_grid_points(nx, ny, dz, zz, points);
		}
		else
		{
			generate_circular_points_hollow(probe_diameter, dz, zz, points);
		}
		zz += dz;
	}

	int n = static_cast<int>(points.size());

	// Compute bounds and lowest z in a single traversal to reduce passes over the data
	float_x lowest_point_z = std::numeric_limits<float_x>::max();
	float_x min_x = std::numeric_limits<float_x>::max();
	float_x max_x = std::numeric_limits<float_x>::lowest();
	float_x min_y = std::numeric_limits<float_x>::max();
	float_x max_y = std::numeric_limits<float_x>::lowest();
	float_x min_z = std::numeric_limits<float_x>::max();
	float_x max_z = std::numeric_limits<float_x>::lowest();

	for (const auto &pt : points)
	{
		if (pt.x < min_x)
			min_x = pt.x;
		if (pt.x > max_x)
			max_x = pt.x;
		if (pt.y < min_y)
			min_y = pt.y;
		if (pt.y > max_y)
			max_y = pt.y;
		if (pt.z < min_z)
			min_z = pt.z;
		if (pt.z > max_z)
			max_z = pt.z;
		if (pt.z < lowest_point_z)
			lowest_point_z = pt.z;
	}

	printf("lowest point z: %f\n", lowest_point_z);
	printf("min_x: %f, max_x: %f, min_y: %f, max_y: %f\n", min_x, max_x, min_y, max_y);

	*grid = new grid_gpu_green(n, make_float3_t(-1. - wp_length / 2. - shift, -1. - wp_width / 2, -1.), make_float3_t(+1. + wp_length / 2. - shift, +1. + wp_width / 2, +2 + wp_thickness + probe_hight), hdx * dz);
	(*grid)->set_bbox_vel(make_float3_t(global_substrate_velocity, 0., global_probe_velocity));
	global_blanking = new blanking(vec3_t(-1. - wp_length / 2 - shift, -1 - wp_width / 2, -1.), vec3_t(+1. + wp_length / 2 - shift, +1. + wp_width / 2, +2 + wp_thickness + probe_hight), vec3_t(global_substrate_velocity, 0., global_probe_velocity), global_wz * global_probe_raduis * global_wz * global_probe_raduis * 2.);
	printf("calculating with %d\n", n);

	float4_t *pos = new float4_t[n];
	float4_t *vel = new float4_t[n];
	float_x *h = new float_x[n];
	float_x *rho = new float_x[n];
	float_x *T = new float_x[n];
	float_x *tool_p = new float_x[n];
	float_x *fixed = new float_x[n];

	for (int i = 0; i < n; i++)
	{
		rho[i] = phys.rho0; 
		tool_p[i] = 1;
		fixed[i] = -2;

		float_x radius = sqrt(points[i].x * points[i].x + points[i].y * points[i].y);
		pos[i] = {points[i].x, points[i].y, points[i].z, i};
		h[i] = hdx * dz;
		vel[i] = {0.0, 0.0, 0.0, 0.0};
		T[i] = joco.Tref;

		// probe 3
		if (pos[i].z >= wp_thickness)
		{
			rho[i] = phys.rho0;
			tool_p[i] = 0;
			fixed[i] = 0;

			glm::vec3 r(pos[i].x, pos[i].y, 0.0);
			glm::vec3 v = glm::cross(w, r);
			vel[i] = {v.x, v.y, global_probe_velocity, 0.0};
			if (pos[i].z >= max_z - 2 * dz)
			{
				fixed[i] = -1; // top surface
			}
		}

		// thermal BC
		if (pos[i].x == min_x || pos[i].x == max_x || pos[i].y == min_y || pos[i].y == max_y)
		{
			fixed[i] = 7;
		}

		if (pos[i].z == 0 || pos[i].z == 0 + dz)
		{
			fixed[i] = 2; // bottom plate
		}
	}

	for (int i = 0; i < n; i++)
	{
		if (tool_p[i] == 1)
		{

			pos[i].x -= shift;

			global_shoulder_contact_surface = std::max(global_shoulder_contact_surface, pos[i].z);
			global_top_surface = global_shoulder_contact_surface;
			global_probe_contact_surface = global_shoulder_contact_surface;
		}
	}

	actions_setup_corrector_constants(corr);
	actions_setup_physical_constants(phys);
	actions_setup_thermal_constants_wp(trml_wp);
	actions_setup_thermal_constants_tool(trml_wp);
	actions_setup_johnson_cook_constants(joco);

	interactions_setup_corrector_constants(corr);
	interactions_setup_physical_constants(phys);
	interactions_setup_geometry_constants(*grid);
	interactions_setup_thermal_constants_workpiece(trml_wp);
	interactions_setup_thermal_constants_tool(trml_wp);
	interactions_setup_johnson_cook_constants(joco);

	particle_gpu *particles = new particle_gpu(pos, vel, rho, T, h, fixed, tool_p, n);

	float_x max_vel = w.z * (probe_diameter / 2.0);
	float_x c0 = sqrt(phys.K / phys.rho0);
	constexpr float_x CFL = 0.3;
	float_x delta_t_max = CFL * hdx * dz / (sqrt(max_vel) + c0);
	global_time_dt = 0.5 * delta_t_max;
	global_time_final = 8. / global_Vsf; // 0.7 s ramping

	global_time_dt_cooling = 1.e-6;
	global_time_cooling = 13.3 / global_Vsf;

	assert(check_cuda_error());
	return particles;
}
