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
check()
{

	vector<DependentField<spacedim-1, spacedim>> dependent_fields(1, DependentField<spacedim-1, spacedim>("q"));

	const double alpha = 1.0;
	const unsigned int method = 0;
	Functions::ConstantFunction<spacedim> constant_fun(0.1);
	const double F = 1.3;
	const double RT = 0.7;
	const double A_e = 0.3;
	const double i_0 = 0.4;
	const double R_el = 0.8;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	OmegaElectrolysis01<spacedim> omega_electrolysis_01(dependent_fields,
														{},
														QGauss<spacedim-1>(1),
														global_data,
														F,
														RT,
														A_e,
														constant_fun,
														i_0,
														R_el,
														method,
														alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	e_omega[0] = 0.4;

	e_omega_ref_sets[0][0] = 0.2;

	Tensor<1, spacedim> n;

	omega_electrolysis_01.compare_derivatives_with_numerical_derivatives(	e_omega,
																			e_omega_ref_sets,
																			hidden_vars,
																			x,
																			n,
																			"derivatives.dat");
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "### 3D-Case ###\n\n";
	check<3>();
}
