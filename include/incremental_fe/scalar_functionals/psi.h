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

#ifndef INCREMENTALFE_SCALARFUNCTIONALS_PSI_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_PSI_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ h^\Sigma_\tau = \alpha \psi(q_{n+1}) + (1-\alpha) \left[ \psi(q_n) + \psi_q(q_n) (q_{n+1} - q_n)\right] \f$,
 *
 * where \f$0\leq \alpha \leq 1 \f$ is a numerical parameter, and \f$q_n\f$ and \f$q_{n+1}\f$ is the value of the state variable \f$q\f$
 * in the beginning of the time step and at the end of the time step, respectively. \f$q\f$ can also be vector valued.
 */
template<unsigned int dim, unsigned int spacedim>
class Psi: public dealii::GalerkinTools::ScalarFunctional<dim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1.
	 */
	const double
	alpha;

public:

	/**
	 * if this is true, ScalarFunctional::get_h_sigma() will always return the value of the potential independent of what alpha is used.
	 */
	bool
	always_compute_potential_value = false;

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					Dependent fields \f$q\f$
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Psi::global_data
	 *
	 * @param[in]		alpha					Psi::alpha
	 *
	 * @param[in]		name					ScalarFunctional::name
	 */
	Psi(	const std::vector<dealii::GalerkinTools::DependentField<dim,spacedim>>	e_sigma,
			const std::set<dealii::types::material_id>								domain_of_integration,
			const dealii::Quadrature<dim>											quadrature,
			GlobalDataIncrementalFE<spacedim>&										global_data,
			const double															alpha = 0.0,
			const std::string														name = "Psi");

	/**
	 * Destructor
	 */
	virtual
	~Psi() = default;

	/**
	 * This function defines \f$\psi(q)\f$ and needs to be implemented by classes inheriting from this class
	 *
	 * @param[in]	values					Values at which \f$\psi\f$ and its derivatives are evaluated
	 *
	 * @param[in]	x						Position
	 *
	 * @param[in]	n						Normal vector
	 *
	 * @param[out]	psi						Value of \f$\psi\f$
	 *
	 * @param[out]	d_psi					Values of first derivatives of \f$\psi\f$ w.r.t. \f$q\f$
	 *
	 * @param[out]	d2_psi					Values of second derivatives of \f$\psi\f$ w.r.t. \f$q\f$
	 *
	 * @param[in]	requested_quantities	Tuple indicating which of the quantities @p psi, @p d_psi, @p d2_psi are to be computed (note that only those quantities
	 * 										are initialized to the correct size, which are actually requested).
	 *
	 * @return								@p true indicates that an error has occurred in the function
	 */
	virtual
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1,spacedim>& 	n,
								double&								psi,
								dealii::Vector<double>&				d_psi,
								dealii::FullMatrix<double>&			d2_psi,
								const std::tuple<bool, bool, bool>	requested_quantities)
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
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ h^\Omega_\rho = \alpha \psi(q_{n+1}) + (1-\alpha) \left[ \psi(q_n) + \psi_q(q_n) (q_{n+1} - q_n)\right] \f$,
 *
 * where \f$0\leq \alpha \leq 1 \f$ is a numerical parameter, and \f$q_n\f$ and \f$q_{n+1}\f$ is the value of the state variable \f$q\f$
 * in the beginning of the time step and at the end of the time step, respectively. \f$q\f$ can also be vector valued.
 */
template<unsigned int spacedim>
class Psi<spacedim, spacedim> : public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1.
	 */
	const double
	alpha;

public:

	/**
	 * if this is true, ScalarFunctional<spacedim, spacedim>::get_h_omega() will always return the value of the potential independent of what alpha is used.
	 */
	bool
	always_compute_potential_value = false;

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					Dependent fields \f$q\f$
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Psi::global_data
	 *
	 * @param[in]		alpha					Psi::alpha
	 *
	 * @param[in]		name					ScalarFunctional::name
	 */
	Psi(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
			const std::set<dealii::types::material_id>									domain_of_integration,
			const dealii::Quadrature<spacedim>											quadrature,
			GlobalDataIncrementalFE<spacedim>&											global_data,
			const double																alpha = 0.0,
			const std::string															name = "Psi");

	/**
	 * Destructor
	 */
	virtual
	~Psi() = default;

	/**
	 * This function defines \f$\psi(q)\f$ and needs to be implemented by classes inheriting from this class
	 *
	 * @param[in]	values					Values at which \f$\psi\f$ and its derivatives are evaluated
	 *
	 * @param[in]	x						Position
	 *
	 * @param[out]	psi						Value of \f$\psi\f$
	 *
	 * @param[out]	d_psi					Values of first derivatives of \f$\psi\f$ w.r.t. \f$q\f$
	 *
	 * @param[out]	d2_psi					Values of second derivatives of \f$\psi\f$ w.r.t. \f$q\f$
	 *
	 * @param[in]	requested_quantities	Tuple indicating which of the quantities @p psi, @p d_psi, @p d2_psi are to be computed (note that only those quantities
	 * 										are initialized to the correct size, which are actually requested).
	 *
	 * @return								@p true indicates that an error has occurred in the function
	 */
	virtual
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								double&								psi,
								dealii::Vector<double>&				d_psi,
								dealii::FullMatrix<double>&			d2_psi,
								const std::tuple<bool, bool, bool>	requested_quantities)
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

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_PSI_H_ */
