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
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/fe_model.h>
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

	const unsigned int method = 0;
	const double alpha = 0.5;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(0.5);
	global_data.set_t(1.5);

	Tensor<1,spacedim> n;
	if(spacedim == 3)
	{
		n[0] = 0.3;
		n[1] = -0.2;
		n[2] = sqrt(1.0 - n[0] * n[0] - n[1] * n[1]);
	}
	else
	{
		n[0] = 0.3;
		n[1] = -sqrt(1.0 - n[0] * n[0]);
	}

	// delta_se_ap
	OmegaDualButlerVolmer00<spacedim> delta_se_ap(	dependent_fields,
													{},
													QGauss<spacedim-1>(1),
													global_data,
													1.3, 0.7, 1.5, 20.0,
													method,
													alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	e_omega[0] = 1.3;
	e_omega_ref_sets[0][0] = -1.3;
	delta_se_ap.compare_derivatives_with_numerical_derivatives(	e_omega,
																e_omega_ref_sets,
																hidden_vars,
																x,
																n);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();
}
