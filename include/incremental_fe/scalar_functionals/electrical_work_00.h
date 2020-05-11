#ifndef INCREMENTALFE_SCALARFUNCTIONALS_ELECTRICALWORK00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_ELECTRICALWORK00_H_

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
 * \f$h^\Sigma_\tau = \tilde\varphi (\boldsymbol{D}^- - \boldsymbol{D}^+) \cdot \boldsymbol{n}\f$,
 *
 * where \f$\boldsymbol{D}\f$ is the dielectric displacement vector, \f$\boldsymbol{n}\f$ the
 * outward unit normal on the interface, and \f$\tilde\varphi\f$ the prescribed scalar potential
 * (which may depend on location and time).
 *
 * Ordering of quantities in ScalarFunctional::e_sigma:<br>	[0] \f$D^-_0 - D^+_0\f$<br>
 * 															[1] \f$D^-_1 - D^+_1\f$<br>
 * 															[2] \f$D^-_2 - D^+_2\f$
 */
template<unsigned int spacedim>
class ElectricalWork00:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * %Function determining \f$\tilde\varphi(t, \boldsymbol{X})\f$
	 * (\f$t\f$ is time and \f$\boldsymbol{X}\f$ the point in space)
	 */
	double
	(*const function_phi)(double, dealii::Point<spacedim>);

	/**
	 * Numerical parameter between @p 0 and @p 1 (see ElectricalWork00::method).
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
				const std::tuple<bool, bool, bool> 			requested_quantities)
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
	 * @param[in]		global_data				ElectricalWork00::global_data
	 *
	 * @param[in] 		function_phi			ElectricalWork00::function_phi
	 *
	 * @param[in]		alpha					ElectricalWork00::alpha
	 *
	 * @param[in]		method					ElectricalWork00::method
	 */
	ElectricalWork00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						double																			(*const function_phi)(double, dealii::Point<spacedim>),
						const double																	alpha = 1.0,
						const unsigned int																method = 1);
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_ELECTRICALWORK00_H_ */
