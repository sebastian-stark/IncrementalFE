#ifndef INCREMENTALFE_SCALARFUNCTIONALS_NEOHOOKE02_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_NEOHOOKE02_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining Neo-Hooke material
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
 */
template<unsigned int spacedim>
class
NeoHooke02:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
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
	 * Numerical parameter between @p 0 and @p 1 (weights derivatives at the beginning and at the end of the load step).
	 */
	const double
	alpha;

	/**
	 * Function allowing to scale lambda and alpha in dependence on position
	 */
	const dealii::Function<spacedim>&
	scaling_function;

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
	 *
	 * @param[in]	x						Position
	 */
	void
	compute_derivatives(const dealii::Vector<double>&		e_omega,
						double&								val,
						dealii::Vector<double>&				d1,
						dealii::FullMatrix<double>& 		d2,
						const std::tuple<bool, bool, bool>&	requested_quantities,
						const dealii::Point<spacedim>& 		x)
	const;


	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	e_omega_ref_sets,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				hidden_vars,
						const dealii::Point<spacedim>& 				x)
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
	 * @param[in]		global_data				NeoHooke02::global_data
	 *
	 * @param[in]		lambda					NeoHooke02::lambda
	 *
	 * @param[in]		mu						NeoHooke02::mu
	 *
	 * @param[in]		alpha					NeoHooke02::alpha
	 *
	 * @param[in]		scaling_function		NeoHooke02::scaling_function
	 */
	NeoHooke02(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																lambda,
						const double																mu,
						const double																alpha,
						const dealii::Function<spacedim>&											scaling_function);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_NEOHOOKE02_H_ */
