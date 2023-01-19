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

	vector<DependentField<spacedim, spacedim>> dependent_fields(12, DependentField<spacedim, spacedim>("q"));

	const double alpha = 1.0;
	const double D = 1.3;
	const double V_m_f = 2.3;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	OmegaFluidDissipation00<spacedim> omega_dual_fluid_dissipation_00(	dependent_fields,
																		{},
																		QGauss<spacedim>(1),
																		global_data,
																		D,
																		V_m_f,
																		method,
																		alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;

	double a = 0.1;
	for(auto& val : e_omega)
	{
		val = cos(a);
		a = a + 0.1;
	}

	for(auto& val : e_omega_ref_sets[0])
	{
		val = sin(a);
		a = a + 0.1;
	}

	e_omega[dependent_fields.size()-9] = 0.82492;
	e_omega[dependent_fields.size()-8] = 0.69239;
	e_omega[dependent_fields.size()-7] = 0.68594;
	e_omega[dependent_fields.size()-6] = 0.72255;
	e_omega[dependent_fields.size()-5] = 0.89152;
	e_omega[dependent_fields.size()-4] = 0.94655;
	e_omega[dependent_fields.size()-3] = 0.39235;
	e_omega[dependent_fields.size()-2] = 0.11984;
	e_omega[dependent_fields.size()-1] = 0.67159;

	e_omega_ref_sets[0][dependent_fields.size()-9] = 0.3926525;
	e_omega_ref_sets[0][dependent_fields.size()-8] = 0.3881020;
	e_omega_ref_sets[0][dependent_fields.size()-7] = 0.3384077;
	e_omega_ref_sets[0][dependent_fields.size()-6] = 0.0096148;
	e_omega_ref_sets[0][dependent_fields.size()-5] = 0.9185749;
	e_omega_ref_sets[0][dependent_fields.size()-4] = 0.2269697;
	e_omega_ref_sets[0][dependent_fields.size()-3] = 0.3312697;
	e_omega_ref_sets[0][dependent_fields.size()-2] = 0.8314307;
	e_omega_ref_sets[0][dependent_fields.size()-1] = 0.9050925;

	omega_dual_fluid_dissipation_00.compare_derivatives_with_numerical_derivatives(	e_omega,
																					e_omega_ref_sets,
																					hidden_vars,
																					x);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>(0);

	cout << "\n### 3D-Case ###\n\n";
	check<3>(0);

	cout << "### 2D-Case ###\n\n";
	check<2>(1);

	cout << "\n### 3D-Case ###\n\n";
	check<3>(1);

}
