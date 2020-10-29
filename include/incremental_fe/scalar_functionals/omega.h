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

#ifndef INCREMENTALFE_SCALARFUNCTIONALS_OMEGA_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_OMEGA_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <galerkin_tools/total_potential_contribution.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining the time discrete approximation of an interface related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma = \omega^\Sigma(\dot{v}, \dot{q}, \mu, q; t) \f$,
 *
 * where \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, and \f$q\f$ may be vectors.
 *
 * The time discrete approximation is done either by Miehe's method, by the \f$\alpha\f$-family, or by the modified \f$\alpha\f$-family.
 */
template<unsigned int dim, unsigned int spacedim>
class Omega: public dealii::GalerkinTools::ScalarFunctional<dim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1.
	 *
	 * This parameter is only used for the \f$\alpha\f$-family and the modified \f$\alpha\f$-family
	 */
	const double
	alpha;

	/**
	 * Temporal discretization
	 * (@p 0:	Miehe's method,
	 *  @p 1: 	\f$\alpha\f$-family,
	 *  @p 2:	modified \f$\alpha\f$-family)
	 */
	const unsigned int
	method;

public:

	/**
	 * Sum of components in \f$\dot{v}\f$ and \f$\dot{q}\f$
	 */
	const unsigned int
	n_v_q_dot;

	/**
	 * Number of Lagrangian multiplier variables
	 */
	const unsigned int
	n_mu;

	/**
	 * Number of state variables
	 */
	const unsigned int
	n_q;

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					Dependent fields (in the order \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, \f$q\f$)
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		n_v_dot					The number of \f$\dot{v}\f$
	 *
	 * @param[in]		n_q_dot					The number of \f$\dot{q}\f$
	 *
	 * @param[in]		n_mu					The number of \f$\mu\f$
	 *
	 * @param[in]		n_q						The number of \f$q\f$
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 *
	 * @param[in]		name					ScalarFunctional::name
	 */
	Omega(	const std::vector<dealii::GalerkinTools::DependentField<dim,spacedim>>	e_sigma,
			const std::set<dealii::types::material_id>								domain_of_integration,
			const dealii::Quadrature<dim>											quadrature,
			GlobalDataIncrementalFE<spacedim>&										global_data,
			const unsigned int														n_v_dot,
			const unsigned int														n_q_dot,
			const unsigned int														n_mu,
			const unsigned int														n_q,
			const unsigned int														method,
			const double															alpha = 0.0,
			const std::string														name = "Omega");

	/**
	 * Destructor
	 */
	virtual
	~Omega() = default;

	/**
	 * This function defines \f$\omega^\Sigma\f$ and needs to be implemented by classes inheriting from this class
	 *
	 * @param[in]	values					Values at which \f$\omega^\Sigma\f$ and its derivatives are evaluated (in the order \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, \f$q\f$)
	 *
	 * @param[in]	t						Time at which \f$\omega^\Sigma\f$ is evaluated
	 *
	 * @param[in]	x						Position
	 *
	 * @param[in]	n						Normal vector
	 *
	 * @param[out]	omega					Value of \f$\omega^\Sigma\f$
	 *
	 * @param[out]	d_omega					Values of first derivatives of \f$\omega^\Sigma\f$ w.r.t. \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$ (in this order)
	 *
	 * @param[out]	d2_omega				Values of second derivatives of \f$\omega^\Sigma\f$ w.r.t. \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$ (in this order). If @p compute_d2q == @p true,
	 * 										also the derivatives of @p d_omega w.r.t. \f$q\f$ need to be computed. In this case, @p d2_omega is initialized to the size
	 * 										Omega::n_v_q_dot + Omega::n_mu x Omega::n_v_q_dot + Omega::n_mu + Omega::n_q, so that derivatives of @p d_omega w.r.t. \f$q\f$ can be stored in
	 * 										the rightmost part of the matrix.
	 *
	 * @param[in]	requested_quantities	Tuple indicating which of the quantities @p omega, @p d_omega, @p d2_omega are to be computed.
	 * 										Note that only those quantities are passed in initialized to the correct size, which are actually requested
	 *
	 * @param[in]	compute_d2q				If @p true, also compute second derivatives w.r.t. \f$q\f$
	 *
	 * @return								@p true indicates that an error has occurred in the function
	 */
	virtual
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1,spacedim>& 	n,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_d2q)
	const = 0;

	/**
	 * see ScalarFunctional::get_h_sigma
	 */
	bool
	get_h_sigma(const dealii::Vector<double>& 				e_sigma,
				const std::vector<dealii::Vector<double>>&	e_sigma_ref_sets,
				dealii::Vector<double>& 					hidden_vars,
				const dealii::Point<spacedim>& 				x,
				const dealii::Tensor<1,spacedim>& 			n,
				double& 									h_sigma,
				dealii::Vector<double>& 					h_sigma_1,
				dealii::FullMatrix<double>& 				h_sigma_2,
				const std::tuple<bool, bool, bool>			requested_quantities)
	const
	final;
};

/**
 * Class defining the time discrete approximation of a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega = \omega^\Omega(\dot{v}, \dot{q}, \mu, q; t) \f$,
 *
 * where \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, and \f$q\f$ may be vectors.
 *
 * The time discrete approximation is done either by Miehe's method, by the \f$\alpha\f$-family, or by the modified \f$\alpha\f$-family.
 */
template<unsigned int spacedim>
class Omega<spacedim, spacedim> : public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1.
	 *
	 * This parameter is only used for the \f$\alpha\f$-family and the modified \f$\alpha\f$-family
	 */
	const double
	alpha;

	/**
	 * Temporal discretization
	 * (@p 0:	Miehe's method,
	 *  @p 1: 	\f$\alpha\f$-family,
	 *  @p 2:	modified \f$\alpha\f$-family)
	 */
	const unsigned int
	method;

public:

	/**
	 * Sum of components in \f$\dot{v}\f$ and \f$\dot{q}\f$
	 */
	const unsigned int
	n_v_q_dot;

	/**
	 * Number of Lagrangian multiplier variables
	 */
	const unsigned int
	n_mu;

	/**
	 * Number of state variables
	 */
	const unsigned int
	n_q;

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					Dependent fields (in the order \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, \f$q\f$)
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		n_v_dot					The number of \f$\dot{v}\f$
	 *
	 * @param[in]		n_q_dot					The number of \f$\dot{q}\f$
	 *
	 * @param[in]		n_mu					The number of \f$\mu\f$
	 *
	 * @param[in]		n_q						The number of \f$q\f$
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 *
	 * @param[in]		name					ScalarFunctional::name
	 */
	Omega(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
			const std::set<dealii::types::material_id>									domain_of_integration,
			const dealii::Quadrature<spacedim>											quadrature,
			GlobalDataIncrementalFE<spacedim>&											global_data,
			const unsigned int															n_v_dot,
			const unsigned int															n_q_dot,
			const unsigned int															n_mu,
			const unsigned int															n_q,
			const unsigned int															method,
			const double																alpha = 0.0,
			const std::string															name = "Omega");

	/**
	 * Destructor
	 */
	virtual
	~Omega() = default;

	/**
	 * This function defines \f$\omega^\Omega\f$ and needs to be implemented by classes inheriting from this class
	 *
	 * @param[in]	values					Values at which \f$\omega^\Omega\f$ and its derivatives are evaluated (in the order \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, \f$q\f$)
	 *
	 * @param[in]	t						Time at which \f$\omega^\Omega\f$ is evaluated
	 *
	 * @param[in]	x						Position
	 *
	 * @param[out]	omega					Value of \f$\omega^\Omega\f$
	 *
	 * @param[out]	d_omega					Values of first derivatives of \f$\omega^\Omega\f$ w.r.t. \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$ (in this order)
	 *
	 * @param[out]	d2_omega				Values of second derivatives of \f$\omega^\Omega\f$ w.r.t. \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$ (in this order). If @p compute_d2q == @p true,
	 * 										also the derivatives of @p d_omega w.r.t. \f$q\f$ need to be computed. In this case, @p d2_omega is initialized to the size
	 * 										Omega::n_v_q_dot + Omega::n_mu x Omega::n_v_q_dot + Omega::n_mu + Omega::n_q, so that derivatives of @p d_omega w.r.t. \f$q\f$ can be stored in
	 * 										the rightmost part of the matrix.
	 *
	 * @param[in]	requested_quantities	Tuple indicating which of the quantities @p omega, @p d_omega, @p d2_omega are to be computed.
	 * 										Note that only those quantities are passed in initialized to the correct size, which are actually requested
	 *
	 * @param[in]	compute_d2q				If @p true, also compute second derivatives w.r.t. \f$q\f$
	 *
	 * @return								@p true indicates that an error has occurred in the function
	 */
	virtual
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								const dealii::Point<spacedim>& 		x,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_d2q)
	const = 0;

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_h_omega
	 */
	bool
	get_h_omega(const dealii::Vector<double>&				e_omega,
				const std::vector<dealii::Vector<double>>&	e_omega_ref_sets,
				dealii::Vector<double>&						hidden_vars,
				const dealii::Point<spacedim>&				x,
				double&										h_omega,
				dealii::Vector<double>&						h_omega_1,
				dealii::FullMatrix<double>&					h_omega_2,
				const std::tuple<bool, bool, bool>			requested_quantities)
	const
	final;

};


/**
 * Class defining the time discrete approximation of a contribution to the incremental potential depending only unknowns not depending upon time
 *
 * \f$ \omega^\mathrm{C} = \omega^\mathrm{C}(\dot{v}, \dot{q}, \mu, q; t) \f$,
 *
 * where \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, and \f$q\f$ may be vectors.
 *
 * The time discrete approximation is done either by Miehe's method, by the \f$\alpha\f$-family, or by the modified \f$\alpha\f$-family.
 *
 * @warning		This class is currently largely untested!
 * @todo		Test this class
 */
template<unsigned int spacedim>
class Omega<0, spacedim> : public dealii::GalerkinTools::TotalPotentialContribution<spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1.
	 *
	 * This parameter is only used for the \f$\alpha\f$-family and the modified \f$\alpha\f$-family
	 */
	const double
	alpha;

	/**
	 * Temporal discretization
	 * (@p 0:	Miehe's method,
	 *  @p 1: 	\f$\alpha\f$-family,
	 *  @p 2:	modified \f$\alpha\f$-family)
	 */
	const unsigned int
	method;

	/**
	 * This vector is used to store the state variables entering the contribution as parameters (only needed for modified \f$\alpha\f$-family)
	 */
	mutable
	dealii::Vector<double>
	state_vars;

	/**
	 * Name of this potential contribution
	 */
	const std::string
	name;

public:

	/**
	 * Sum of components in \f$\dot{v}\f$ and \f$\dot{q}\f$
	 */
	const unsigned int
	n_v_q_dot;

	/**
	 * Number of Lagrangian multiplier variables
	 */
	const unsigned int
	n_mu;

	/**
	 * Number of state variables
	 */
	const unsigned int
	n_q;

	/**
	 * Constructor
	 *
	 * @param[in]		C						Unknowns entering the potential contribution (in the order \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, \f$q\f$)
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		n_v_dot					The number of \f$\dot{v}\f$
	 *
	 * @param[in]		n_q_dot					The number of \f$\dot{q}\f$
	 *
	 * @param[in]		n_mu					The number of \f$\mu\f$
	 *
	 * @param[in]		n_q						The number of \f$q\f$
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 *
	 * @param[in]		name					ScalarFunctional::name
	 */
	Omega(	const std::vector<const dealii::GalerkinTools::IndependentField<0, spacedim>*>	C,
			GlobalDataIncrementalFE<spacedim>&												global_data,
			const unsigned int																n_v_dot,
			const unsigned int																n_q_dot,
			const unsigned int																n_mu,
			const unsigned int																n_q,
			const unsigned int																method,
			const double																	alpha = 0.0,
			const std::string																name = "Omega");

	/**
	 * Destructor
	 */
	virtual
	~Omega() = default;

	/**
	 * This function defines \f$\omega^\mathrm{C}\f$ and needs to be implemented by classes inheriting from this class
	 *
	 * @param[in]	values					Values at which \f$\omega^\mathrm{C}\f$ and its derivatives are evaluated (in the order \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$, \f$q\f$)
	 *
	 * @param[in]	t						Time at which \f$\omega^\mathrm{C}\f$ is evaluated
	 *
	 * @param[out]	omega					Value of \f$\omega^\mathrm{C}\f$
	 *
	 * @param[out]	d_omega					Values of first derivatives of \f$\omega^\mathrm{C}\f$ w.r.t. \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$ (in this order)
	 *
	 * @param[out]	d2_omega				Values of second derivatives of \f$\omega^\mathrm{C}\f$ w.r.t. \f$\dot{v}\f$, \f$\dot{q}\f$, \f$\mu\f$ (in this order). If @p compute_d2q == @p true,
	 * 										also the derivatives of @p d_omega w.r.t. \f$q\f$ need to be computed. In this case, @p d2_omega is initialized to the size
	 * 										Omega::n_v_q_dot + Omega::n_mu x Omega::n_v_q_dot + Omega::n_mu + Omega::n_q, so that derivatives of @p d_omega w.r.t. \f$q\f$ can be stored in
	 * 										the rightmost part of the matrix.
	 *
	 * @param[in]	requested_quantities	Tuple indicating which of the quantities @p omega, @p d_omega, @p d2_omega are to be computed.
	 * 										Note that only those quantities are passed in initialized to the correct size, which are actually requested
	 *
	 * @param[in]	compute_d2q				If @p true, also compute second derivatives w.r.t. \f$q\f$
	 *
	 * @return								@p true indicates that an error has occurred in the function
	 */
	virtual
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_d2q)
	const = 0;

	/**
	 * see TotalPotentialContribution:get_potential_contribution
	 */
	bool
	get_potential_contribution( const dealii::Vector<double>&				H_omega_H_sigma_C,
								const std::vector<dealii::Vector<double>>&	C_ref_sets,
								double&										Pi,
								dealii::Vector<double>&						Pi_1,
								dealii::FullMatrix<double>&					Pi_2,
								const std::tuple<bool,bool,bool>&			requested_quantities)
	const
	final;

};


}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_OMEGA_H_ */
