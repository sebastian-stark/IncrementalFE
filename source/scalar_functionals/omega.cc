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

#include <incremental_fe/scalar_functionals/omega.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
Omega<spacedim, spacedim>::Omega(	const vector<DependentField<spacedim,spacedim>>	e_omega,
									const set<types::material_id>					domain_of_integration,
									const Quadrature<spacedim>						quadrature,
									GlobalDataIncrementalFE<spacedim>&				global_data,
									const unsigned int								n_v_dot,
									const unsigned int								n_q_dot,
									const unsigned int								n_mu,
									const unsigned int								n_q,
									const unsigned int								method,
									const double									alpha,
									const string									name)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, name, 1, method == 2 ? n_q : 0),
global_data(&global_data),
alpha(alpha),
method(method),
n_v_q_dot(n_v_dot + n_q_dot),
n_mu(n_mu),
n_q(n_q)
{
	Assert(	e_omega.size() == (n_v_dot + n_q_dot + n_mu + n_q),
			ExcMessage("The number of dependent fields passed does not coincide with n_v_dot + n_q_dot + n_mu + n_q !") );
	Assert( (method < 3),
			ExcMessage("You requested a temporal integration method, which is not implemented !"));
	Assert( (alpha >= 0.0) && (alpha <= 1.0),
			ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));

	if(method == 2)
		global_data.set_predictor_corrector();
}

template<unsigned int spacedim>
bool
Omega<spacedim, spacedim>::get_h_omega(	Vector<double>&					e_omega,
										const vector<Vector<double>>&	e_omega_ref_sets,
										Vector<double>&					hidden_vars,
										const Point<spacedim>&			x,
										double&							h_omega,
										Vector<double>&					h_omega_1,
										FullMatrix<double>&				h_omega_2,
										const tuple<bool, bool, bool>	requested_quantities)
const
{
	// times
	const double dt = global_data->get_t() - global_data->get_t_ref();

	// Vector in which the approximated values of q_dot, mu, q are sorted
	Vector<double> values(n_v_q_dot + n_mu + n_q);

	// time of evaluation
	double t = 0.0;

	for(unsigned int m = 0; m < n_v_q_dot; ++m)
		values[m] = (e_omega[m] - e_omega_ref_sets[0][m])/dt;
	for(unsigned int m = n_v_q_dot; m < n_v_q_dot + n_mu; ++m)
		values[m] = e_omega[m];
	if(method == 0)
	{
		t = global_data->get_t();
		for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			values[m] = e_omega_ref_sets[0][m];
	}
	else if(method == 1)
	{
		t = (1.0 - alpha) * global_data->get_t_ref() + alpha * global_data->get_t();
		for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			values[m] = (1.0 - alpha) * e_omega_ref_sets[0][m] + alpha * e_omega[m];
	}
	else if(method == 2)
	{
		t = (1.0 - alpha) * global_data->get_t_ref() + alpha * global_data->get_t();
		if(global_data->get_predictor_step())
		{
			for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			{
				values[m] = e_omega_ref_sets[0][m];
				// make sure that predicted values are stored
				hidden_vars[m - n_v_q_dot - n_mu] = e_omega[m];
			}
		}
		else
		{
			for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
				values[m] = (1.0 - alpha) * e_omega_ref_sets[0][m] + alpha * hidden_vars[m - n_v_q_dot - n_mu];
		}
	}

	double omega;
	Vector<double> d_omega;
	FullMatrix<double> d2_omega;
	if(get<1>(requested_quantities))
		d_omega.reinit(n_v_q_dot + n_mu);
	if(get<2>(requested_quantities))
		d2_omega.reinit(n_v_q_dot + n_mu, method == 1 ? n_v_q_dot + n_mu + n_q : n_v_q_dot + n_mu);

	//get derivatives
	if(get_values_and_derivatives(values, t, x, omega, d_omega, d2_omega, requested_quantities, method == 1 ? true : false))
		return true;

	// sort into return quantities
	if(get<0>(requested_quantities))
	{
		if((method != 1) && compute_potential_value)
			h_omega = dt * omega;
		else
			h_omega = 0.0;
	}
	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(e_omega.size());
		for(unsigned int m = 0; m < n_v_q_dot + n_mu; ++m)
			h_omega_1[m] = dt * d_omega[m];
		for(unsigned int m = 0; m < n_v_q_dot; ++m)
			h_omega_1[m] *= 1.0/dt;
	}
	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(e_omega.size(), e_omega.size());
		for(unsigned int m = 0; m < d2_omega.m(); ++m)
			for(unsigned int n = 0; n < d2_omega.n(); ++n)
				h_omega_2(m, n) = dt * d2_omega(m,n);

		for(unsigned int m = 0; m < n_v_q_dot; ++m)
		{
			for(unsigned int n = 0; n < n_v_q_dot + n_mu + n_q; ++n)
			{
				h_omega_2(m, n) *= 1.0/dt;
				h_omega_2(n, m) *= 1.0/dt;
			}
		}

		if(method == 1)
		{
			for(unsigned int m = 0; m < n_v_q_dot + n_mu; ++m)
				for(unsigned int n = n_v_q_dot + n_mu; n < n_v_q_dot + n_mu + n_q; ++n)
					h_omega_2(m, n) *= alpha;
		}
	}

	return false;
}

template<unsigned int dim, unsigned int spacedim>
Omega<dim,spacedim>::Omega(	const vector<DependentField<dim,spacedim>>	e_sigma,
							const set<types::material_id>				domain_of_integration,
							const Quadrature<dim>						quadrature,
							GlobalDataIncrementalFE<spacedim>&			global_data,
							const unsigned int							n_v_dot,
							const unsigned int							n_q_dot,
							const unsigned int							n_mu,
							const unsigned int							n_q,
							const unsigned int							method,
							const double								alpha,
							const string								name)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, name, 1, method == 2 ? n_q : 0),
global_data(&global_data),
alpha(alpha),
method(method),
n_v_q_dot(n_v_dot + n_q_dot),
n_mu(n_mu),
n_q(n_q)
{
	Assert(	e_sigma.size() == (n_v_dot + n_q_dot + n_mu + n_q),
			ExcMessage("The number of dependent fields passed does not coincide with n_v_dot + n_q_dot + n_mu + n_q !") );
	Assert( (method < 3),
			ExcMessage("You requested a temporal integration method, which is not implemented !"));
	Assert( (alpha >= 0.0) && (alpha <= 1.0),
			ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
	if(method == 2)
		global_data.set_predictor_corrector();
}

template<unsigned int dim, unsigned int spacedim>
bool
Omega<dim, spacedim>::get_h_sigma(	Vector<double>& 				e_sigma,
									const vector<Vector<double>>&	e_sigma_ref_sets,
									Vector<double>& 				hidden_vars,
									const Point<spacedim>& 			x,
									const Tensor<1,spacedim>& 		n,
									double& 						h_sigma,
									Vector<double>& 				h_sigma_1,
									FullMatrix<double>& 			h_sigma_2,
									const tuple<bool, bool, bool>	requested_quantities)
const
{
	// times
	const double dt = global_data->get_t() - global_data->get_t_ref();

	// Vector in which the approximated values of q_dot, mu, q are sorted
	Vector<double> values(n_v_q_dot + n_mu + n_q);

	// time of evaluation
	double t = 0.0;

	for(unsigned int m = 0; m < n_v_q_dot; ++m)
		values[m] = (e_sigma[m] - e_sigma_ref_sets[0][m])/dt;
	for(unsigned int m = n_v_q_dot; m < n_v_q_dot + n_mu; ++m)
		values[m] = e_sigma[m];
	if(method == 0)
	{
		t = global_data->get_t();
		for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			values[m] = e_sigma_ref_sets[0][m];
	}
	else if(method == 1)
	{
		t = (1.0 - alpha) * global_data->get_t_ref() + alpha * global_data->get_t();
		for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			values[m] = (1.0 - alpha) * e_sigma_ref_sets[0][m] + alpha * e_sigma[m];
	}
	else if(method == 2)
	{
		t = (1.0 - alpha) * global_data->get_t_ref() + alpha * global_data->get_t();
		if(global_data->get_predictor_step())
		{
			for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			{
				values[m] = e_sigma_ref_sets[0][m];
				// make sure that predicted values are stored
				hidden_vars[m - n_v_q_dot - n_mu] = e_sigma[m];
			}
		}
		else
		{
			for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
				values[m] = (1.0 - alpha) * e_sigma_ref_sets[0][m] + alpha * hidden_vars[m - n_v_q_dot - n_mu];
		}
	}

	double sigma;
	Vector<double> d_sigma;
	FullMatrix<double> d2_sigma;
	if(get<1>(requested_quantities))
		d_sigma.reinit(n_v_q_dot + n_mu);
	if(get<2>(requested_quantities))
		d2_sigma.reinit(n_v_q_dot + n_mu, method == 1 ? n_v_q_dot + n_mu + n_q : n_v_q_dot + n_mu);

	//get derivatives
	if(get_values_and_derivatives(values, t, x, n, sigma, d_sigma, d2_sigma, requested_quantities, method == 1 ? true : false))
		return true;

	// sort into return quantities
	if(get<0>(requested_quantities))
	{
		if((method != 1) && compute_potential_value)
			h_sigma = dt * sigma;
		else
			h_sigma = 0.0;
	}
	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(e_sigma.size());
		for(unsigned int m = 0; m < n_v_q_dot + n_mu; ++m)
			h_sigma_1[m] = dt * d_sigma[m];
		for(unsigned int m = 0; m < n_v_q_dot; ++m)
			h_sigma_1[m] *= 1.0/dt;
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(e_sigma.size(), e_sigma.size());
		for(unsigned int m = 0; m < d2_sigma.m(); ++m)
			for(unsigned int n = 0; n < d2_sigma.n(); ++n)
				h_sigma_2(m, n) = dt * d2_sigma(m,n);

		for(unsigned int m = 0; m < n_v_q_dot; ++m)
		{
			for(unsigned int n = 0; n < n_v_q_dot + n_mu + n_q; ++n)
			{
				h_sigma_2(m, n) *= 1.0/dt;
				h_sigma_2(n, m) *= 1.0/dt;
			}
		}

		if(method == 1)
		{
			for(unsigned int m = 0; m < n_v_q_dot + n_mu; ++m)
				for(unsigned int n = n_v_q_dot + n_mu; n < n_v_q_dot + n_mu + n_q; ++n)
					h_sigma_2(m, n) *= alpha;
		}
	}

	return false;
}

template<unsigned int spacedim>
Omega<0, spacedim>::Omega(	const std::vector<const dealii::GalerkinTools::IndependentField<0, spacedim>*>	C,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const unsigned int																n_v_dot,
							const unsigned int																n_q_dot,
							const unsigned int																n_mu,
							const unsigned int																n_q,
							const unsigned int																method,
							const double																	alpha,
							const std::string																name)
:
TotalPotentialContribution<spacedim>(std::vector<const ScalarFunctional<spacedim, spacedim>*>(), std::vector<const ScalarFunctional<spacedim-1, spacedim>*>(), C),
global_data(&global_data),
alpha(alpha),
method(method),
state_vars(dealii::Vector<double>(method == 2 ? n_q : 0)),
name(name),
n_v_q_dot(n_v_dot + n_q_dot),
n_mu(n_mu),
n_q(n_q)
{
	Assert(	C.size() == (n_v_dot + n_q_dot + n_mu + n_q),
			ExcMessage("The number of unknowns passed does not coincide with n_v_dot + n_q_dot + n_mu + n_q !") );
	Assert( (method < 3),
			ExcMessage("You requested a temporal integration method, which is not implemented !"));
	Assert( (alpha >= 0.0) && (alpha <= 1.0),
			ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));

	if(method == 2)
		global_data.set_predictor_corrector();
}


template<unsigned int spacedim>
bool
Omega<0, spacedim>::get_potential_contribution(	const Vector<double>&			C,
												const vector<Vector<double>>&	C_ref_sets,
												double&							Pi,
												dealii::Vector<double>&			Pi_1,
												dealii::FullMatrix<double>&		Pi_2,
												const tuple<bool,bool,bool>&	requested_quantities)
const
{
	// times
	const double dt = global_data->get_t() - global_data->get_t_ref();

	// Vector in which the approximated values of q_dot, mu, q are sorted
	Vector<double> values(n_v_q_dot + n_mu + n_q);

	// time of evaluation
	double t = 0.0;

	for(unsigned int m = 0; m < n_v_q_dot; ++m)
		values[m] = (C[m] - C_ref_sets[0][m])/dt;
	for(unsigned int m = n_v_q_dot; m < n_v_q_dot + n_mu; ++m)
		values[m] = C[m];
	if(method == 0)
	{
		t = global_data->get_t();
		for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			values[m] = C_ref_sets[0][m];
	}
	else if(method == 1)
	{
		t = (1.0 - alpha) * global_data->get_t_ref() + alpha * global_data->get_t();
		for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			values[m] = (1.0 - alpha) * C_ref_sets[0][m] + alpha * C[m];
	}
	else if(method == 2)
	{
		t = (1.0 - alpha) * global_data->get_t_ref() + alpha * global_data->get_t();
		if(global_data->get_predictor_step())
		{
			for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
			{
				values[m] = C_ref_sets[0][m];
				// make sure that predicted values are stored
				state_vars[m - n_v_q_dot - n_mu] = C[m];
			}
		}
		else
		{
			for(unsigned int m = n_v_q_dot + n_mu; m < n_v_q_dot + n_mu + n_q; ++m)
				values[m] = (1.0 - alpha) * C_ref_sets[0][m] + alpha * state_vars[m - n_v_q_dot - n_mu];
		}
	}

	double omega;
	Vector<double> d_omega;
	FullMatrix<double> d2_omega;
	if(get<1>(requested_quantities))
		d_omega.reinit(n_v_q_dot + n_mu);
	if(get<2>(requested_quantities))
		d2_omega.reinit(n_v_q_dot + n_mu, method == 1 ? n_v_q_dot + n_mu + n_q : n_v_q_dot + n_mu);

	//get derivatives
	if(get_values_and_derivatives(values, t, omega, d_omega, d2_omega, requested_quantities, method == 1 ? true : false))
		return true;

	// sort into return quantities
	if(get<0>(requested_quantities))
	{
		if((method != 1) && compute_potential_value)
			Pi = dt * omega;
		else
			Pi = 0.0;
	}
	if(get<1>(requested_quantities))
	{
		Pi_1.reinit(C.size());
		for(unsigned int m = 0; m < n_v_q_dot + n_mu; ++m)
			Pi_1[m] = dt * d_omega[m];
		for(unsigned int m = 0; m < n_v_q_dot; ++m)
			Pi_1[m] *= 1.0/dt;
	}
	if(get<2>(requested_quantities))
	{
		Pi_2.reinit(C.size(), C.size());
		for(unsigned int m = 0; m < d2_omega.m(); ++m)
			for(unsigned int n = 0; n < d2_omega.n(); ++n)
				Pi_2(m, n) = dt * d2_omega(m,n);

		for(unsigned int m = 0; m < n_v_q_dot; ++m)
		{
			for(unsigned int n = 0; n < n_v_q_dot + n_mu + n_q; ++n)
			{
				Pi_2(m, n) *= 1.0/dt;
				Pi_2(n, m) *= 1.0/dt;
			}
		}

		if(method == 1)
		{
			for(unsigned int m = 0; m < n_v_q_dot + n_mu; ++m)
				for(unsigned int n = n_v_q_dot + n_mu; n < n_v_q_dot + n_mu + n_q; ++n)
					Pi_2(m, n) *= alpha;
		}
	}

	return false;
}


template class incrementalFE::Omega<2,2>;
template class incrementalFE::Omega<3,3>;
template class incrementalFE::Omega<1,2>;
template class incrementalFE::Omega<2,3>;
template class incrementalFE::Omega<0,2>;
template class incrementalFE::Omega<0,3>;
