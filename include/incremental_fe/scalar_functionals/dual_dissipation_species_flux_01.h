#ifndef INCREMENTALFE_SCALARFUNCTIONALS_DUALDISSIPATIONSPECIESFLUX01_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_DUALDISSIPATIONSPECIESFLUX01_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining dissipation associated with species flux in large deformation setting.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0] \f$\vareta_{,x}\f$<br>
 * 																				[1] \f$\vareta_{,y}\f$<br>
 * 																				[2] \f$\vareta_{,z}\f$<br>
 * 																				[3] \f$c\f$<br>
 * 																				[4] \f$\eta\f$<br>
 * 																				[5] \f$F_xx\f$<br>
 * 																				[6] \f$F_xy\f$<br>
 * 																				[7] \f$F_xz\f$<br>
 * 																				[8] \f$F_yx\f$<br>
 * 																				[9] \f$F_yy\f$<br>
 * 																				[10] \f$F_yz\f$<br>
 * 																				[11] \f$F_zx\f$<br>
 * 																				[12] \f$F_zy\f$<br>
 * 																				[13] \f$F_zz\f$<br>
 */
template<unsigned int spacedim>
class
DualDissipationSpeciesFlux01:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
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
	 * @param[in]		global_data				DissipationSpeciedFlux01::global_data
	 *
	 * @param[in]		D						DissipationSpeciedFlux01::D
	 *
	 * @param[in]		method					DualDissipationSpeciesFlux01::method
	 */
	DualDissipationSpeciesFlux01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>									domain_of_integration,
									const dealii::Quadrature<spacedim>											quadrature,
									GlobalDataIncrementalFE<spacedim>&											global_data,
									const double																D,
									const unsigned int															method);
};

}




#endif /* INCREMENTALFE_SCALARFUNCTIONALS_DUALDISSIPATIONSPECIESFLUX01_H_ */
