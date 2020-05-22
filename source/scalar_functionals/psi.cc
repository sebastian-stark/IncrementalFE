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

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/psi.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
Psi<spacedim, spacedim>::Psi(	const vector<DependentField<spacedim,spacedim>>	e_omega,
								const set<types::material_id>					domain_of_integration,
								const Quadrature<spacedim>						quadrature,
								GlobalDataIncrementalFE<spacedim>&				global_data,
								const double									alpha,
								const string									name)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, name, 1),
global_data(&global_data),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0),
			ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
Psi<spacedim, spacedim>::get_h_omega(	const Vector<double>&			e_omega,
										const vector<Vector<double>>&	e_omega_ref_sets,
										Vector<double>&					/*hidden_vars*/,
										const Point<spacedim>&			x,
										double&							h_omega,
										Vector<double>&					h_omega_1,
										FullMatrix<double>&				h_omega_2,
										const tuple<bool, bool, bool>	requested_quantities)
const
{
	//compute derivatives at alpha = 1.0
	if(get<1>(requested_quantities))
		h_omega_1.reinit(e_omega.size());
	if(get<2>(requested_quantities))
		h_omega_2.reinit(e_omega.size(), e_omega.size());

	if(get_values_and_derivatives(e_omega, x, h_omega, h_omega_1, h_omega_2, requested_quantities))
		return true;

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		double h_omega_ref;
		Vector<double>	h_omega_1_ref;
		FullMatrix<double>	h_omega_2_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			h_omega_1_ref.reinit(e_omega.size());
			if(get_values_and_derivatives(e_omega_ref_sets[0], x, h_omega_ref, h_omega_1_ref, h_omega_2_ref, make_tuple(true, true, false)))
				return true;
		}

		if(get<0>(requested_quantities))
			h_omega = h_omega * alpha + ( h_omega_ref + h_omega_1_ref * e_omega - h_omega_1_ref * e_omega_ref_sets[0] ) * (1. - alpha);

		if(get<1>(requested_quantities))
			for(unsigned int m = 0; m < e_omega.size(); ++m)
				h_omega_1[m] = h_omega_1[m] * alpha + h_omega_1_ref[m] * (1. - alpha);

		if(get<2>(requested_quantities))
			h_omega_2 *= alpha;
	}

	return false;
}

template<unsigned int dim, unsigned int spacedim>
Psi<dim,spacedim>::Psi(	const vector<DependentField<dim,spacedim>>	e_sigma,
						const set<types::material_id>				domain_of_integration,
						const Quadrature<dim>						quadrature,
						GlobalDataIncrementalFE<spacedim>&			global_data,
						const double								alpha,
						const string								name)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, name, 1),
global_data(&global_data),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0),
			ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int dim, unsigned int spacedim>
bool
Psi<dim, spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
									const vector<Vector<double>>&	e_sigma_ref_sets,
									Vector<double>& 				/*hidden_vars*/,
									const Point<spacedim>& 			x,
									const Tensor<1,spacedim>& 		n,
									double& 						h_sigma,
									Vector<double>& 				h_sigma_1,
									FullMatrix<double>& 			h_sigma_2,
									const tuple<bool, bool, bool>	requested_quantities)
const
{
	//compute derivatives at alpha = 1.0
	if(get<1>(requested_quantities))
		h_sigma_1.reinit(e_sigma.size());
	if(get<2>(requested_quantities))
		h_sigma_2.reinit(e_sigma.size(), e_sigma.size());

	if(get_values_and_derivatives(e_sigma, x, n, h_sigma, h_sigma_1, h_sigma_2, requested_quantities))
		return true;

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		double h_sigma_ref;
		Vector<double>	h_sigma_1_ref;
		FullMatrix<double>	h_sigma_2_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			h_sigma_1_ref.reinit(e_sigma.size());
			if(get_values_and_derivatives(e_sigma_ref_sets[0], x, n, h_sigma_ref, h_sigma_1_ref, h_sigma_2_ref, make_tuple(false, true, false)))
				return true;
		}

		if(get<0>(requested_quantities))
			h_sigma = h_sigma * alpha + ( h_sigma_ref + h_sigma_1_ref * e_sigma - h_sigma_1_ref * e_sigma_ref_sets[0] ) * (1. - alpha);

		if(get<1>(requested_quantities))
			for(unsigned int m = 0; m < e_sigma.size(); ++m)
				h_sigma_1[m] = h_sigma_1[m] * alpha + h_sigma_1_ref[m] * (1. - alpha);

		if(get<2>(requested_quantities))
			h_sigma_2 *= alpha;
	}

	return false;
}

template class Psi<2,2>;
template class Psi<3,3>;
template class Psi<1,2>;
template class Psi<2,3>;
