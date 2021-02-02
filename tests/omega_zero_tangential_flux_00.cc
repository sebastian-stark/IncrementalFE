// --------------------------------------------------------------------------
// Copyright (C) 2020 by Sebastian Stark
//
// This file is part of the IncrementalFE library
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include <iostream>
#include <time.h>
#include <stdlib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/omega_lib.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check(unsigned int method)
{

	vector<DependentField<spacedim-1, spacedim>> dependent_fields(18, DependentField<spacedim-1, spacedim>("q"));

	const double alpha = 1.0;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	OmegaZeroTangentialFlux00<spacedim> omega_zero_tangential_flux(	dependent_fields,
																	{},
																	QGauss<spacedim-1>(1),
																	global_data,
																	false,
																	method,
																	alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	e_omega[0] = 0.82492;
	e_omega[1] = 0.69239;
	e_omega[2] = 0.68594;
	e_omega[3] = 0.72255;
	e_omega[4] = 0.89152;
	e_omega[5] = 0.94655;
	e_omega[6] = 0.39235;
	e_omega[7] = 0.11984;
	e_omega[8] = 0.67159;
	e_omega[9] = 0.82492;
	e_omega[10] = 0.69239;
	e_omega[11] = 0.68594;
	e_omega[12] = 0.72255;
	e_omega[13] = 0.89152;
	e_omega[14] = 0.94655;
	e_omega[15] = 0.39235;
	e_omega[16] = 0.11984;
	e_omega[17] = 0.67159;

	e_omega_ref_sets[0][0] = 0.3926525;
	e_omega_ref_sets[0][1] = 0.3881020;
	e_omega_ref_sets[0][2] = 0.3384077;
	e_omega_ref_sets[0][3] = 0.0096148;
	e_omega_ref_sets[0][4] = 0.9185749;
	e_omega_ref_sets[0][5] = 0.2269697;
	e_omega_ref_sets[0][6] = 0.3312697;
	e_omega_ref_sets[0][7] = 0.8314307;
	e_omega_ref_sets[0][8] = 0.9050925;
	e_omega_ref_sets[0][9] = 0.3926525;
	e_omega_ref_sets[0][10] = 0.3881020;
	e_omega_ref_sets[0][11] = 0.3384077;
	e_omega_ref_sets[0][12] = 0.0096148;
	e_omega_ref_sets[0][13] = 0.9185749;
	e_omega_ref_sets[0][14] = 0.2269697;
	e_omega_ref_sets[0][15] = 0.3312697;
	e_omega_ref_sets[0][16] = 0.8314307;
	e_omega_ref_sets[0][17] = 0.9050925;

	Tensor<1, spacedim> n;
	n[0] = 0.3;
	n[1] = 0.2;
	if(spacedim == 3)
		n[2] = sqrt(1.0 - n[0]*n[0] - n[1]*n[1]);

	omega_zero_tangential_flux.compare_derivatives_with_numerical_derivatives(	e_omega,
																				e_omega_ref_sets,
																				hidden_vars,
																				x,
																				n,
																				"derivatives.dat");
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>(0);

	cout << "### 3D-Case ###\n\n";
	check<3>(0);

	cout << "### 2D-Case ###\n\n";
	check<2>(1);

	cout << "### 3D-Case ###\n\n";
	check<3>(1);

}
