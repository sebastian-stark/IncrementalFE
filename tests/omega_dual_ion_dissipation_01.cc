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

	vector<DependentField<spacedim, spacedim>> dependent_fields(15, DependentField<spacedim, spacedim>("q"));

	const double alpha = 1.0;
	const double D = 1.3;
	const double n_0 = 0.25;
	const double V_m_f = 0.7;
	const unsigned int method = 0;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	OmegaDualIonDissipation01<spacedim> omega_dual_fluid_dissipation_01(	dependent_fields,
																			{},
																			QGauss<spacedim>(1),
																			global_data,
																			D,
																			n_0,
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

	const unsigned int i_F = 6;
	e_omega_ref_sets[0][i_F + 0] = 0.17400;
	e_omega_ref_sets[0][i_F + 1] = 0.65967;
	e_omega_ref_sets[0][i_F + 2] = 0.55345;
	e_omega_ref_sets[0][i_F + 3] = 0.96550;
	e_omega_ref_sets[0][i_F + 4] = 0.29811;
	e_omega_ref_sets[0][i_F + 5] = 0.73196;
	e_omega_ref_sets[0][i_F + 6] = 0.72997;
	e_omega_ref_sets[0][i_F + 7] = 0.79393;
	e_omega_ref_sets[0][i_F + 8] = 0.10799;

	omega_dual_fluid_dissipation_01.compare_derivatives_with_numerical_derivatives(	e_omega,
																					e_omega_ref_sets,
																					hidden_vars,
																					x);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();

}
