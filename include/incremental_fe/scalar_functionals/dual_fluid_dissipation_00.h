#ifndef INCREMENTALFE_SCALARFUNCTIONALS_DUALFLUIDDISSIPATION00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_DUALFLUIDDISSIPATION00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining dual dissipation associated with flux of fluid through solid skeleton.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\eta^\mathrm{F}_{,x}\f$<br>
 * 																				[1]  \f$\eta^\mathrm{F}_{,y}\f$<br>
 * 																				[2]  \f$\eta^\mathrm{F}_{,z}\f$<br>
 * 																				[3]  \f$\eta^\mathrm{+}_{,x}\f$<br>
 * 																				[4]  \f$\eta^\mathrm{+}_{,y}\f$<br>
 * 																				[5]  \f$\eta^\mathrm{+}_{,z}\f$<br>
 * 																				[6]  \f$\eta^\mathrm{-}_{,x}\f$<br>
 * 																				[7]  \f$\eta^\mathrm{-}_{,y}\f$<br>
 * 																				[8]  \f$\eta^\mathrm{-}_{,z}\f$<br>
 * 																				[9]  \f$\eta^\mathrm{F}\f$<br>
 * 																				[10] \f$\eta^\mathrm{+}\f$<br>
 * 																				[11] \f$\eta^\mathrm{-}\f$<br>
 * 																				[12] \f$c^\mathrm{F}\f$<br>
 * 																				[13] \f$c^\mathrm{+}\f$<br>
 * 																				[14] \f$c^\mathrm{-}\f$<br>
 * 																				[15] \f$F_xx\f$<br>
 * 																				[16] \f$F_xy\f$<br>
 * 																				[17] \f$F_xz\f$<br>
 * 																				[18] \f$F_yx\f$<br>
 * 																				[19] \f$F_yy\f$<br>
 * 																				[20] \f$F_yz\f$<br>
 * 																				[21] \f$F_zx\f$<br>
 * 																				[22] \f$F_zy\f$<br>
 * 																				[23] \f$F_zz\f$<br>
 */
template<unsigned int spacedim>
class
DualFluidDissipation00:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Diffusion coefficient of species \f$D\f$
	 */
	const double
	D;

	/**
	 * molar volume of fluid \f$V^\mathrm{F}_\mathrm{m}\f$
	 */
	const double
	V_m_F;

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
	 * @param[in]		global_data				DualFluidDissipation00::global_data
	 *
	 * @param[in]		D						DualFluidDissipation00::D
	 *
	 * @param[in]		V_m_F					DualFluidDissipation00::V_m_F
	 *
	 * @param[in]		method					DualFluidDissipation00::method
	 */
	DualFluidDissipation00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const double																D,
							const double																V_m_F,
							const unsigned int															method);
};

}




#endif /* INCREMENTALFE_SCALARFUNCTIONALS_DUALFLUIDDISSIPATION00_H_ */
