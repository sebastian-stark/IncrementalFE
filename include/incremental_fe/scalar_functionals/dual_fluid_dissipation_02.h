#ifndef INCREMENTALFE_SCALARFUNCTIONALS_DUALFLUIDDISSIPATION02_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_DUALFLUIDDISSIPATION02_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining dual dissipation associated with flux of fluid
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\lambda_{x}\f$<br>
 * 																				[1]  \f$\lambda_{y}\f$<br>
 * 																				[2]  \f$\lambda_{z}\f$<br>
 * 																				[3]  \f$\eta^\mathrm{F}_{,x}\f$<br>
 * 																				[4]  \f$\eta^\mathrm{F}_{,y}\f$<br>
 * 																				[5]  \f$\eta^\mathrm{F}_{,z}\f$<br>
 * 																				[6]  \f$\eta^\mathrm{+}_{,x}\f$<br>
 * 																				[7]  \f$\eta^\mathrm{+}_{,y}\f$<br>
 * 																				[8]  \f$\eta^\mathrm{+}_{,z}\f$<br>
 * 																				[9]  \f$\eta^\mathrm{-}_{,x}\f$<br>
 * 																				[10] \f$\eta^\mathrm{-}_{,y}\f$<br>
 * 																				[11] \f$\eta^\mathrm{-}_{,z}\f$<br>
 * 																				[12] \f$\eta^\mathrm{F}\f$<br>
 * 																				[13] \f$c^\mathrm{F}\f$<br>
 * 																				[14] \f$c^\mathrm{+}\f$<br>
 * 																				[15] \f$c^\mathrm{-}\f$
 */
template<unsigned int spacedim>
class
DualFluidDissipation02:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

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
	 * @param[in]		global_data				DualFluidDissipation02::global_data
	 *
	 * @param[in]		method					DualFluidDissipation02::method
	 */
	DualFluidDissipation02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const unsigned int															method);
};

}




#endif /* INCREMENTALFE_SCALARFUNCTIONALS_DUALFLUIDDISSIPATION02_H_ */
