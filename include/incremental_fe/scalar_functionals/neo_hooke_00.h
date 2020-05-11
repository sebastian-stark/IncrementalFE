#ifndef INCREMENTALFE_SCALARFUNCTIONALS_NEOHOOKE00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_NEOHOOKE00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining Neo-Hooke material with purely volumetric inelastic strain.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$F_xx\f$<br>
 * 																					[1] \f$F_xy\f$<br>
 * 																					[2] \f$F_xz\f$<br>
 * 																					[3] \f$F_yx\f$<br>
 * 																					[4] \f$F_yy\f$<br>
 * 																					[5] \f$F_yz\f$<br>
 * 																					[6] \f$F_zx\f$<br>
 * 																					[7] \f$F_zy\f$<br>
 * 																					[8] \f$F_zz\f$<br>
 * 																					[9] \f$c\f$
 */
template<unsigned int spacedim>
class
NeoHooke00:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Lame constant
	 */
	const double
	lambda;

	/**
	 * Lame constant
	 */
	const double
	mu;

	/**
	 * Maximum concentration of species
	 */
	const double
	c_max;

	/**
	 * Linear strain at maximum concentration (referred to 0 concentration)
	 */
	const double
	eps;

	/**
	 * species concentration in reference state at the beginning of the computation (this state is assumed to be stress-free)
	 */
	const double
	c_0;


	/**
	 * Numerical parameter between @p 0 and @p 1 (weights derivatives at the beginning and at the end of the load step).
	 */
	const double
	alpha;

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_h_omega
	 */
	bool
	get_h_omega(const dealii::Vector<double>& 				e_omega,
				const std::vector<dealii::Vector<double>>&	e_omega_ref_sets,
				dealii::Vector<double>&						hidden_vars,
				const dealii::Point<spacedim>&				x,
				double&										h_omega,
				dealii::Vector<double>&						h_omega_1,
				dealii::FullMatrix<double>&					h_omega_2,
				const std::tuple<bool, bool, bool>			requested_quantities)
	const;

	/**
	 * Function computing the value of the continuous potential and its derivatives
	 *
	 * @param[in]	e_omega					The values of the dependent variables
	 *
	 * @param[out]	val						The value of the potential
	 *
	 * @param[out]	d1						First derivatives
	 *
	 * @param[out]	d2						Second derivatives
	 *
	 * @param[in]	requested_quantities	Indicates which quantities are to be computed
	 */
	void
	compute_derivatives(const dealii::Vector<double>&		e_omega,
						double&								val,
						dealii::Vector<double>&				d1,
						dealii::FullMatrix<double>& 		d2,
						const std::tuple<bool, bool, bool>&	requested_quantities)
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
	 * @param[in]		global_data				NeoHooke00::global_data
	 *
	 * @param[in]		lambda					NeoHooke00::lambda
	 *
	 * @param[in]		mu						NeoHooke00::mu
	 *
	 * @param[in]		c_max					NeoHooke00::c_max
	 *
	 * @param[in]		eps						NeoHooke00::eps
	 *
	 * @param[in]		c_0						NeoHooke00::c_0
	 *
	 * @param[in]		alpha					NeoHooke00::alpha
	 */
	NeoHooke00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																lambda,
						const double																mu,
						const double																c_max,
						const double																eps,
						const double																c_0,
						const double																alpha);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_NEOHOOKE00_H_ */
