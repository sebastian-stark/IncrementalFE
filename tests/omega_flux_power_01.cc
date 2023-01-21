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

//Function for prescribing external chemical potential due to external solution
template <unsigned int spacedim>
class MuF : public Function<spacedim>
{
private:

public:

	double
	value(	const Point<spacedim>& /*p*/,
			const unsigned int		/*component=0*/)
	const
	{
		return 1.3;
	}
};

template<unsigned int spacedim>
void
check(unsigned int method)
{

	vector<DependentField<spacedim-1, spacedim>> dependent_fields(1, DependentField<spacedim-1, spacedim>("q"));

	const double alpha = 1.0;

	MuF<spacedim> fun;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	OmegaFluxPower01<spacedim> omega_flux_power_01(	dependent_fields,
													{},
													QGauss<spacedim-1>(1),
													global_data,
													fun,
													method,
													alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	e_omega[0] = 0.82492;

	e_omega_ref_sets[0][0] = 0.3926525;

	Tensor<1, spacedim> n;
	n[0] = 0.3;
	n[1] = 0.2;
	if(spacedim == 3)
		n[2] = sqrt(1.0 - n[0]*n[0] - n[1]*n[1]);

	omega_flux_power_01.compare_derivatives_with_numerical_derivatives(	e_omega,
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
