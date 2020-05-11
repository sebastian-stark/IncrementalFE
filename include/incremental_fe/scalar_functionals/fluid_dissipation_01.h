#ifndef INCREMENTALFE_SCALARFUNCTIONALS_FLUIDDISSIPATION01_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_FLUIDDISSIPATION01_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining dissipation associated with viscosity of a fluid.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$u_{x,x}\f$<br>
 * 																				[1]  \f$u_{x,y}\f$<br>
 * 																				[2]  \f$u_{x,z}\f$<br>
 * 																				[3]  \f$u_{y,x}\f$<br>
 * 																				[4]  \f$u_{y,y}\f$<br>
 * 																				[5]  \f$u_{y,z}\f$<br>
 * 																				[6]  \f$u_{z,x}\f$<br>
 * 																				[7]  \f$u_{z,y}\f$<br>
 * 																				[8]  \f$u_{z,z}\f$<br>
 * 																				[9]  \f$F_xx\f$<br>
 * 																				[10] \f$F_xy\f$<br>
 * 																				[11] \f$F_xz\f$<br>
 * 																				[12] \f$F_yx\f$<br>
 * 																				[13] \f$F_yy\f$<br>
 * 																				[14] \f$F_yz\f$<br>
 * 																				[15] \f$F_zx\f$<br>
 * 																				[16] \f$F_zy\f$<br>
 * 																				[17] \f$F_zz\f$<br>
 */
template<unsigned int spacedim>
class
FluidDissipation01:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Viscosity \f$\eta\f$
	 */
	const double
	eta;

	/**
	 *
	 * If @p method==0: Miehe's method (first order accurate)
	 * If @p method==1: Miehe's method as prediction step + correction step (second order accurate)
	 */
	const unsigned int
	method;

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_h_omega
	 */
	bool
	get_h_omega(	const dealii::Vector<double>&				e_omega,
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
	 * @param[in]		global_data				FluidDissipation01::global_data
	 *
	 * @param[in]		eta						FluidDissipation01::eta
	 *
	 * @param[in]		method					FluidDissipation01::method
	 */
	FluidDissipation01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																eta,
						const unsigned int															method);
};

}




#endif /* INCREMENTALFE_SCALARFUNCTIONALS_FLUIDDISSIPATION01_H_ */
