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
using namespace CMF;

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
Psi<spacedim, spacedim>::get_h_omega(	Vector<double>&					e_omega,
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

	eval_time = global_data->get_t();
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
			eval_time = global_data->get_t_ref();
			if(get_values_and_derivatives(e_omega_ref_sets[0], x, h_omega_ref, h_omega_1_ref, h_omega_2_ref, make_tuple(true, true, false)))
				return true;
		}

		if(get<0>(requested_quantities))
		{
			if(!always_compute_potential_value)
				h_omega = h_omega * alpha + ( h_omega_ref + h_omega_1_ref * e_omega - h_omega_1_ref * e_omega_ref_sets[0] ) * (1. - alpha);
		}

		if(get<1>(requested_quantities))
			for(unsigned int m = 0; m < e_omega.size(); ++m)
				h_omega_1[m] = h_omega_1[m] * alpha + h_omega_1_ref[m] * (1. - alpha);

		if(get<2>(requested_quantities))
			h_omega_2 *= alpha;
	}

	return false;
}

template<unsigned int spacedim>
double
Psi<spacedim,spacedim>::get_eval_time()
const
{
	return eval_time;
}

template<unsigned int spacedim>
void
Psi<spacedim,spacedim>::set_alpha(const double alpha)
{
	this->alpha = alpha;
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
Psi<dim, spacedim>::get_h_sigma(	Vector<double>& 				e_sigma,
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

	eval_time = global_data->get_t();
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
			eval_time = global_data->get_t_ref();
			if(get_values_and_derivatives(e_sigma_ref_sets[0], x, n, h_sigma_ref, h_sigma_1_ref, h_sigma_2_ref, make_tuple(false, true, false)))
				return true;
		}

		if(get<0>(requested_quantities))
		{
			if(!always_compute_potential_value)
				h_sigma = h_sigma * alpha + ( h_sigma_ref + h_sigma_1_ref * e_sigma - h_sigma_1_ref * e_sigma_ref_sets[0] ) * (1. - alpha);
		}

		if(get<1>(requested_quantities))
			for(unsigned int m = 0; m < e_sigma.size(); ++m)
				h_sigma_1[m] = h_sigma_1[m] * alpha + h_sigma_1_ref[m] * (1. - alpha);

		if(get<2>(requested_quantities))
			h_sigma_2 *= alpha;
	}

	return false;
}

template<unsigned int dim, unsigned int spacedim>
double
Psi<dim,spacedim>::get_eval_time()
const
{
	return eval_time;
}

template<unsigned int dim, unsigned int spacedim>
void
Psi<dim,spacedim>::set_alpha(const double alpha)
{
	this->alpha = alpha;
}

#ifdef INCREMENTAL_FE_WITH_CMF

template<unsigned int dim, unsigned int spacedim>
PsiWrapperCMF<dim,spacedim>::PsiWrapperCMF(	ScalarFunction<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>& 	psi,
											const vector<DependentField<dim,spacedim>>														e_sigma,
											const set<types::material_id>																	domain_of_integration,
											const Quadrature<dim>																			quadrature,
											GlobalDataIncrementalFE<spacedim>&																global_data,
											const double																					alpha,
											const string																					name,
											const bool																						use_param,
											dealii::Function<spacedim> *const																param_fun)
:
Psi<dim, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, alpha, name),
psi(psi),
use_param(use_param),
param_fun(param_fun)
{
#ifdef DEBUG
	const unsigned int N_params_x_n = use_param ? 6 : 0;
	const unsigned int N_params_fun = param_fun ? param_fun->n_components : 0;

	Assert( (psi.get_N_parameters() >= N_params_x_n + N_params_fun),
			ExcMessage("The function psi does not have enough parameters to store the position vector, the normal vector and the extra parameters."));
#endif
}

template<unsigned int dim, unsigned int spacedim>
bool
PsiWrapperCMF<dim,spacedim>::get_values_and_derivatives(const Vector<double>& 			values,
														const Point<spacedim>& 			x,
														const Tensor<1,spacedim>& 		n,
														double&							psi_,
														Vector<double>&					d_psi,
														FullMatrix<double>&				d2_psi,
														const tuple<bool, bool, bool>	requested_quantities)
const
{
	Eigen::VectorXd gradient;
	Eigen::MatrixXd hessian;

	Eigen::VectorXd arguments(values.size());

	for(unsigned int m = 0; m < values.size(); ++m)
		arguments(m) = values(m);
	if(psi.set_arguments(arguments))
		return true;

	auto parameters = psi.get_parameters();
	if(use_param)
	{
		for(unsigned int m = 0; m < spacedim; ++m)
			parameters(m) = x[m];
		if(spacedim == 2)
			parameters(2) = 0.0;
		for(unsigned int m = 0; m < spacedim; ++m)
			parameters(m+3) = n[m];
		if(spacedim == 2)
			parameters(5) = 0.0;
	}
	if(param_fun)
	{
		const double t_ = param_fun->get_time();
		param_fun->set_time(this->get_eval_time());
		const unsigned int start_index = use_param ? 6 : 0;
		for(unsigned int m = 0; m < param_fun->n_components; ++m)
			parameters(start_index + m) = param_fun->value(x, m);
		param_fun->set_time(t_);
	}
	if(use_param || param_fun)
	{
		if(psi.set_parameters(parameters))
			return true;
	}

	if(psi.compute(psi_, gradient, hessian, get<0>(requested_quantities), get<1>(requested_quantities), get<2>(requested_quantities)))
		return true;

	if(get<1>(requested_quantities))
	{
		for(unsigned int m = 0; m < d_psi.size(); ++m)
			d_psi(m) = gradient(m);
	}

	if(get<2>(requested_quantities))
	{
		for(unsigned int m = 0; m < d2_psi.m(); ++m)
			for(unsigned int n = 0; n < d2_psi.n(); ++n)
				d2_psi(m,n) = hessian(m,n);
	}

	return false;
}

template<unsigned int dim, unsigned int spacedim>
double
PsiWrapperCMF<dim,spacedim>::get_maximum_step(	const Vector<double>& 					e_omega,
												const vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
												const Vector<double>& 					delta_e_omega,
												const Vector<double>& 					/*hidden_vars*/,
												const Point<spacedim>& 					/*x*/,
												const Tensor<1, spacedim>&				/*n*/)
const
{
	double factor = 2.0;
	Eigen::VectorXd e(e_omega.size());

	while(true)
	{
		for(unsigned int m = 0; m < e.size(); ++m)
			e[m] = e_omega[m] + factor * delta_e_omega[m];

		if(psi.is_in_domain(e))
			return factor;

		factor *= 0.5;
		Assert(factor > 0.0, dealii::ExcMessage("Cannot determine a positive scaling of the load step such that the resulting state is admissible!"));
	}

	return factor;
}

template<unsigned int spacedim>
PsiWrapperCMF<spacedim,spacedim>::PsiWrapperCMF(ScalarFunction<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>& 	psi,
												const vector<DependentField<spacedim,spacedim>>													e_omega,
												const set<types::material_id>																	domain_of_integration,
												const Quadrature<spacedim>																		quadrature,
												GlobalDataIncrementalFE<spacedim>&																global_data,
												const double																					alpha,
												const string																					name,
												const bool																						use_param,
												dealii::Function<spacedim> *const																param_fun)
:
Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, name),
psi(psi),
use_param(use_param),
param_fun(param_fun)
{
#ifdef DEBUG
	const unsigned int N_params_x = use_param ? 3 : 0;
	const unsigned int N_params_fun = param_fun ? param_fun->n_components : 0;

	Assert( (psi.get_N_parameters() >= N_params_x + N_params_fun),
			ExcMessage("The function psi does not have enough parameters to store the position vector, the normal vector and the extra parameters."));
#endif
}

template<unsigned int spacedim>
bool
PsiWrapperCMF<spacedim,spacedim>::get_values_and_derivatives(	const Vector<double>& 			values,
																const Point<spacedim>& 			x,
																double&							psi_,
																Vector<double>&					d_psi,
																FullMatrix<double>&				d2_psi,
																const tuple<bool, bool, bool>	requested_quantities)
const
{
	Eigen::VectorXd gradient;
	Eigen::MatrixXd hessian;

	Eigen::VectorXd arguments(values.size());
	for(unsigned int m = 0; m < values.size(); ++m)
		arguments(m) = values(m);
	if(psi.set_arguments(arguments))
		return true;

	auto parameters = psi.get_parameters();
	if(use_param)
	{
		for(unsigned int m = 0; m < spacedim; ++m)
			parameters(m) = x[m];
		if(spacedim == 2)
			parameters(2) = 0.0;
	}
	if(param_fun)
	{
		const double t_ = param_fun->get_time();
		param_fun->set_time(this->get_eval_time());
		const unsigned int start_index = use_param ? 3 : 0;
		for(unsigned int m = 0; m < param_fun->n_components; ++m)
			parameters(start_index + m) = param_fun->value(x, m);
		param_fun->set_time(t_);
	}
	if(use_param || param_fun)
	{
		if(psi.set_parameters(parameters))
			return true;
	}

	if(psi.compute(psi_, gradient, hessian, get<0>(requested_quantities), get<1>(requested_quantities), get<2>(requested_quantities)))
		return true;

	if(get<1>(requested_quantities))
	{
		for(unsigned int m = 0; m < d_psi.size(); ++m)
			d_psi(m) = gradient(m);
	}

	if(get<2>(requested_quantities))
	{
		for(unsigned int m = 0; m < d2_psi.m(); ++m)
			for(unsigned int n = 0; n < d2_psi.n(); ++n)
				d2_psi(m,n) = hessian(m,n);
	}

	return false;
}

template<unsigned int spacedim>
double
PsiWrapperCMF<spacedim,spacedim>::get_maximum_step(	const Vector<double>& 					e_omega,
													const vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
													const Vector<double>& 					delta_e_omega,
													const Vector<double>& 					/*hidden_vars*/,
													const Point<spacedim>& 					/*x*/)
const
{
	double factor = 2.0;
	Eigen::VectorXd e(e_omega.size());

	while(true)
	{
		for(unsigned int m = 0; m < e.size(); ++m)
			e[m] = e_omega[m] + factor * delta_e_omega[m];

		if(psi.is_in_domain(e))
			return factor;

		factor *= 0.5;
		Assert(factor > 0.0, dealii::ExcMessage("Cannot determine a positive scaling of the load step such that the resulting state is admissible!"));
	}

	return factor;
}

#endif /* INCREMENTAL_FE_WITH_CMF */


template class incrementalFE::Psi<2,2>;
template class incrementalFE::Psi<3,3>;
template class incrementalFE::Psi<1,2>;
template class incrementalFE::Psi<2,3>;
template class incrementalFE::PsiWrapperCMF<2,2>;
template class incrementalFE::PsiWrapperCMF<3,3>;
template class incrementalFE::PsiWrapperCMF<1,2>;
template class incrementalFE::PsiWrapperCMF<2,3>;
