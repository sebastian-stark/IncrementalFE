#ifndef INCREMENTALFE_SCALARFUNCTIONALS_RATEOFDEPENDENTVARIABLES00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_RATEOFDEPENDENTVARIABLES00_H_

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
 * 	\f$h^\Sigma_\tau = (e^\Sigma_\nu-e^\Sigma_{\nu,\mathrm{ref}})/\Delta t\f$,
 *
 * 	where \f$e^\Sigma_\nu\f$ is the dependent field at the end of the time step,
 * 	\f$e^\Sigma_{\nu,\mathrm{ref}}\f$ is the dependent field in the beginning of the time step,
 * 	and \f$\Delta t\f$ is the length of the time step.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma:<br>	[0] \f$e^\Sigma_\nu\f$
 */
template<unsigned int dim, unsigned int spacedim>
class RateOfDependentVariables00 : public dealii::GalerkinTools::ScalarFunctional<dim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * see ScalarFunctional::get_h_sigma
	 */
	bool
	get_h_sigma(const dealii::Vector<double>& 				e_sigma,
				const std::vector<dealii::Vector<double>>&	e_sigma_ref,
				dealii::Vector<double>& 					hidden_vars,
				const dealii::Point<spacedim>& 				x,
				const dealii::Tensor<1,spacedim>& 			n,
				double& 									h_sigma,
				dealii::Vector<double>& 					h_sigma_1,
				dealii::FullMatrix<double>& 				h_sigma_2,
				const std::tuple<bool, bool, bool>			requested_quantities)
	const;

public:

	/**
	 * Constructor
	 *
	 * @param[in]	e_sigma					ScalarFunctional::e_sigma
	 *
	 * @param[in] 	domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]	quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]	global_data				RateOfDependentVariables00::global_data
	 */
	RateOfDependentVariables00(	dealii::GalerkinTools::DependentField<dim, spacedim>	e_sigma,
								const std::set<dealii::types::material_id>				domain_of_integration,
								const dealii::Quadrature<dim>							quadrature,
								GlobalDataIncrementalFE<spacedim>&						global_data);
};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * 	\f$h^\Omega_\rho = (e^\Omega_\lambda-e^\Omega_{\lambda,\mathrm{ref}})/\Delta t\f$,
 *
 * 	where \f$e^\Omega_\lambda\f$ is the dependent field at the end of the time step,
 * 	\f$e^\Omega_{\lambda,\mathrm{ref}}\f$ is the dependent field in the beginning of the time step,
 * 	and \f$\Delta t\f$ is the length of the time step.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega:<br>	[0] \f$e^\Omega_\lambda\f$
 */
template<unsigned int spacedim>
class RateOfDependentVariables00<spacedim, spacedim> : public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

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
	const;

public:

	/**
	 * Constructor
	 *
	 * @param[in]	e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 	domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]	quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]	global_data				RateOfDependentVariables00<spacedim, spacedim>::global_data
	 */
	RateOfDependentVariables00(	dealii::GalerkinTools::DependentField<spacedim,spacedim>	e_omega,
								const std::set<dealii::types::material_id>					domain_of_integration,
								const dealii::Quadrature<spacedim>							quadrature,
								GlobalDataIncrementalFE<spacedim>&							global_data);
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_RATEOFDEPENDENTVARIABLES00_H_ */
