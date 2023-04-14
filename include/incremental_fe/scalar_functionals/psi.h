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
#include <incremental_fe/config.h>
#include <cmf/scalar_function.h>

#include <string>

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
protected:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

private:

	/**
	 * Numerical parameter between @p 0 and @p 1.
	 */
	double
	alpha;

	/**
	 * The time at which the next call of ScalarFunctional::get_h_sigma is performed
	 */
	mutable double
	eval_time = 0.0;

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
	get_h_sigma(dealii::Vector<double>& 					e_sigma,
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

	/**
	 * @return 	The time at which the next call of ScalarFunctional::get_h_sigma is performed.
	 */
	double
	get_eval_time()
	const;

	/**
	 * @param[in]	alpha	value to set for Psi::alpha
	 */
	void
	set_alpha(const double alpha);

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

protected:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

private:

	/**
	 * Numerical parameter between @p 0 and @p 1.
	 */
	double
	alpha;

	/**
	 * The time at which the next call of ScalarFunctional::get_h_omega is performed
	 */
	mutable double
	eval_time = 0.0;

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
	get_h_omega(dealii::Vector<double>&						e_omega,
				const std::vector<dealii::Vector<double>>&	e_omega_ref_sets,
				dealii::Vector<double>&						hidden_vars,
				const dealii::Point<spacedim>&				x,
				double&										h_omega,
				dealii::Vector<double>&						h_omega_1,
				dealii::FullMatrix<double>&					h_omega_2,
				const std::tuple<bool, bool, bool>			requested_quantities)
	const
	final;

	/**
	 * @return 	The time at which the next call of ScalarFunctional::get_h_omega is performed.
	 */
	double
	get_eval_time()
	const;

	/**
	 * @param[in]	alpha	value to set for Psi<spacedim, spacedim>::alpha
	 */
	void
	set_alpha(const double alpha);

};

#ifdef INCREMENTAL_FE_WITH_CMF

/**
 * Interface Free Energy wrapping a function defined with the CMF library.
 *
 * If PsiWrapperCMF::use_param is true, it is assumed that the first six parameters of the CMF function are as follows:<br>
 *
 * parameters[0]   ... parameters[2]	- position vector x<br>
 * parameters[3]   ... parameters[5]	- normal vector n<br>
 *
 * In two dimensions, the third components of x and n are filled with zeroes.
 *
 * Further parameters may follow and are defined through PsiWrapperCMF::param_fun.
 *
 * @warning	This class is untested so far!
 */
template<unsigned int dim, unsigned int spacedim>
class PsiWrapperCMF: public incrementalFE::Psi<dim, spacedim>
{

private:

	/**
	 * The function wrapped
	 */
	CMF::ScalarFunction<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>&
	psi;

	/**
	 * Whether to use position vector and normal vector as parameters
	 */
	const bool
	use_param = true;

	/**
	 * Parameter function (corresponding parameters are appended to parameter vector)
	 */
	dealii::Function<spacedim> *const
	param_fun;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		psi						The function to be wrapped
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
	 *
	 * @param[in]		use_param				PsiWrapperCMF<spacedim,spacedim>::use_param
	 *
	 * @param[in]		param_fun				PsiWrapperCMF<spacedim,spacedim>::param_fun
	 */
	PsiWrapperCMF(	CMF::ScalarFunction<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>& 	psi,
					const std::vector<dealii::GalerkinTools::DependentField<dim,spacedim>>								e_sigma,
					const std::set<dealii::types::material_id>															domain_of_integration,
					const dealii::Quadrature<dim>																		quadrature,
					GlobalDataIncrementalFE<spacedim>&																	global_data,
					const double																						alpha,
					const std::string																					name,
					const bool																							use_param = true,
					dealii::Function<spacedim> *const																	param_fun = nullptr);

	/**
	 * This function defines \f$\psi(q)\f$ based on the wrapped function.
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
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1,spacedim>& 	n,
								double&								psi,
								dealii::Vector<double>&				d_psi,
								dealii::FullMatrix<double>&			d2_psi,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	override
	final;

	/**
	 * @see ScalarFunctional::get_maximum_step()
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_sigma,
						const std::vector<dealii::Vector<double>>&	e_sigma_ref_sets,
						const dealii::Vector<double>& 				delta_e_sigma,
						const dealii::Vector<double>&				hidden_vars,
						const dealii::Point<spacedim>&				x,
						const dealii::Tensor<1, spacedim>&			n)
	const
	override
	final;

};

/**
 * Volume Free Energy wrapping a function defined with the CMF library
 *
 * If PsiWrapperCMF<spacedim,spacedim>::use_param is true, it is assumed that the first three parameters of the CMF function are as follows:<br>
 *
 * parameters[0]          ... parameters[2]	- position vector x<br>
 *
 * In two dimensions, the third component of x is set to zero.
 *
 * Further parameters may follow and are defined through PsiWrapperCMF<spacedim,spacedim>::param_fun.
 *
 * @warning	This class is untested so far!
 */
template<unsigned int spacedim>
class PsiWrapperCMF<spacedim,spacedim> : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * The function wrapped
	 */
	CMF::ScalarFunction<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>&
	psi;

	/**
	 * Whether to use position vector as parameter
	 */
	const bool
	use_param = true;

	/**
	 * Parameter function (corresponding parameters are appended to parameter vector)
	 */
	dealii::Function<spacedim> *const
	param_fun = nullptr;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		psi						The function to be wrapped
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
	 *
	 * @param[in]		use_param				PsiWrapperCMF<spacedim,spacedim>::use_param
	 *
	 * @param[in]		param_fun				PsiWrapperCMF<spacedim,spacedim>::param_fun
	 */
	PsiWrapperCMF(	CMF::ScalarFunction<double, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>& 	psi,
					const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>							e_omega,
					const std::set<dealii::types::material_id>															domain_of_integration,
					const dealii::Quadrature<spacedim>																	quadrature,
					GlobalDataIncrementalFE<spacedim>&																	global_data,
					const double																						alpha,
					const std::string																					name,
					const bool																							use_param = true,
					dealii::Function<spacedim> *const																	param_fun = nullptr);

	/**
	 * This function defines \f$\psi(q)\f$ based on the wrapped function.
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
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								double&								psi,
								dealii::Vector<double>&				d_psi,
								dealii::FullMatrix<double>&			d2_psi,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	override
	final;

	/**
	 * @see ScalarFunctional<spacedim,spacedim>::get_maximum_step()
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	e_omega_ref_sets,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				hidden_vars,
						const dealii::Point<spacedim>& 				x)
	const
	override
	final;

};

#endif /* INCREMENTAL_FE_WITH_CMF */

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_PSI_H_ */
