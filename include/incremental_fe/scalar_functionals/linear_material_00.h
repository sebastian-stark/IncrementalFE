#ifndef INCREMENTALFE_SCALARFUNCTIONALS_LINEARMATERIAL00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_LINEARMATERIAL00_H_

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
 * \f$h^\Sigma_\tau = \dfrac{1}{2} {\boldsymbol{e}^\Sigma}^\top \cdot \boldsymbol{C} \cdot \boldsymbol{e}^\Sigma + \boldsymbol{y}^\top \cdot \boldsymbol{e}^\Sigma\f$,
 *
 * where the vector \f$\boldsymbol{e}^\Sigma\f$ contains a subset of the dependent fields \f$e^\Sigma_\nu\f$, and the matrix \f$\boldsymbol{C}\f$ as well as the vector \f$\boldsymbol{y}\f$ are material specific.
 *
 */
template<unsigned int dim, unsigned int spacedim>
class LinearMaterial00: public dealii::GalerkinTools::ScalarFunctional<dim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * \f$\boldsymbol{C}\f$
	 */
	const dealii::FullMatrix<double>
	C;

	/**
	 * \f$\boldsymbol{y}\f$
	 */
	const dealii::Vector<double>
	y;

	/**
	 * Numerical parameter between @p 0 and @p 1 (see LinearMaterial00::method).
	 *
	 * @warning In the very first time step @p alpha=1.0 is used irrespective of the
	 * 			value provided for @p alpha! This helps to avoid oscillations of the
	 * 			solution if the initial solution fields are not equilibrated.
	 */
	const double
	alpha;

	/**
	 * Numerical method
	 * (@p 0:	take derivatives of scalar functional at time corresponding to @p alpha,
	 *  @p 1: 	weight derivatives at the beginning of the time step and at the end of
	 *  		the time step according to @p alpha)
	 */
	const unsigned
	int method;

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
	 * @param[in]		e_sigma					ScalarFunctional::e_sigma
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				LinearMaterial00::global_data
	 *
	 * @param[in] 		C						LinearMaterial00::C
	 *
	 * @param[in]		y						LinearMaterial00::y
	 *
	 * @param[in]		name					ScalarFunctional::name
	 *
	 * @param[in]		alpha					LinearMaterial00::alpha
	 *
	 * @param[in]		method					LinearMaterial00::method
	 */
	LinearMaterial00(	const std::vector<dealii::GalerkinTools::DependentField<dim,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>								domain_of_integration,
						const dealii::Quadrature<dim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&										global_data,
						const dealii::FullMatrix<double>										C,
						const dealii::Vector<double>											y,
						const std::string														name = "",
						const double															alpha = 1.0,
						const unsigned int														method = 1);
};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = \dfrac{1}{2} {\boldsymbol{e}^\Omega}^\top \cdot \boldsymbol{C} \cdot \boldsymbol{e}^\Omega + \boldsymbol{y}^\top \cdot \boldsymbol{e}^\Omega\f$,
 *
 * where the vector \f$\boldsymbol{e}^\Omega\f$ contains a subset of the dependent fields \f$e^\Omega_\lambda\f$, and the matrix \f$\boldsymbol{C}\f$ as well as the vector \f$\boldsymbol{y}\f$ are material specific.
 *
 */
template<unsigned int spacedim>
class LinearMaterial00<spacedim, spacedim> : public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * \f$\boldsymbol{C}\f$
	 */
	const dealii::FullMatrix<double>
	C;

	/**
	 * \f$\boldsymbol{y}\f$
	 */
	const dealii::Vector<double>
	y;

	/**
	 * Numerical parameter between @p 0 and @p 1 (see LinearMaterial00<spacedim, spacedim>::method).
	 *
	 * @warning In the very first time step @p alpha=1.0 is used irrespective of the
	 * 			value provided for @p alpha! This helps to avoid oscillations of the
	 * 			solution if the initial solution fields are not equilibrated.
	 */
	const double
	alpha;

	/**
	 * Numerical method
	 * (@p 0:	take derivatives of scalar functional at time corresponding to @p alpha,
	 *  @p 1: 	weight derivatives at the beginning of the time step and at the end of
	 *  		the time step according to @p alpha)
	 */
	const unsigned int
	method;

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
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				LinearMaterial00<spacedim, spacedim>::global_data
	 *
	 * @param[in] 		C						LinearMaterial00<spacedim, spacedim>::C
	 *
	 * @param[in]		y						LinearMaterial00<spacedim, spacedim>::y
	 *
	 * @param[in]		name					ScalarFunctional<spacedim, spacedim>::name
	 *
	 * @param[in]		alpha					LinearMaterial00<spacedim, spacedim>::alpha
	 *
	 * @param[in]		method					LinearMaterial00<spacedim, spacedim>::method
	 */
	LinearMaterial00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const dealii::FullMatrix<double>											C,
						const dealii::Vector<double>												y,
						const std::string															name = "",
						const double																alpha = 1.0,
						const unsigned int															method = 1);
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_LINEARMATERIAL00_H_ */
