#ifndef INCREMENTALFE_SCALARFUNCTIONALS_PENALTY00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_PENALTY00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_rho = -\mu \ln\dfrac{c}{c_0}\f$,
 *
 * where \f$c\f$ is a species concentration, \f$c_0\f$ is a reference species concentration,
 * and \f$\mu\f$ is the penalty parameter.
 *
 * This defines the standard logarithmic penalty function enforcing \f$c>0\f$, which could
 * e.g. be used in the context of the interior point method.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega:<br>	[0] \f$c\f$
 */
template<unsigned int spacedim>
class
Penalty00:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Penalty parameter \f$\mu\f$
	 */
	const double
	mu;

	/**
	 * Reference concentration \f$c>0\f$
	 */
	const double
	c0;

	/**
	 * Numerical parameter between @p 0 and @p 1 (see Penalty00::method).
	 *
	 * This value is only used if the interior point method is NOT activated by
	 * Penalty00::global_data. If it is activated by Penalty00::global_data,
	 * the value provided by global_data will be used.
	 *
	 * @warning In the very first time step @p alpha=1.0 is used irrespective of the
	 * 			value provided for @p alpha! This helps to avoid oscillations of the
	 * 			solution if the initial solution fields are not equilibrated.
	 */
	const
	double
	alpha;

	/**
	 * Numerical method
	 * (@p 0:	take derivatives of scalar functional at time corresponding to @p alpha,
	 *  @p 1: 	weight derivatives at the beginning of the time step and at the end of
	 *  		the time step according to @p alpha)
	 */
	const unsigned int method;

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_h_omega
	 */
	bool get_h_omega(	const dealii::Vector<double>&				e_omega,
						const std::vector<dealii::Vector<double>>&	e_omega_ref,
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
	 * @param[in]		global_data				Penalty00::global_data
	 *
	 * @param[in]		mu						Penalty00::mu
	 *
	 * @param[in]		c0						Penalty00::c0
	 *
	 * @param[in]		alpha					Penalty00::alpha
	 *
	 * @param[in]		method					Penalty00::method
	 */
	Penalty00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
				const std::set<dealii::types::material_id>									domain_of_integration,
				const dealii::Quadrature<spacedim>											quadrature,
				GlobalDataIncrementalFE<spacedim>&											global_data,
				const double																mu,
				const double																c0,
				const double																alpha,
				const unsigned																method = 1);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_PENALTY00_H_ */
