#ifndef INCREMENTALFE_SCALARFUNCTIONALS_INTERFACEDISSIPATION02_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_INTERFACEDISSIPATION02_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 *
 * Class defining the Butler-Volmer relation on interfaces in large deformation
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$I_x\f$<br>
 * 															[1] \f$I_y\f$<br>
 * 															[2] \f$I_z\f$<br>
 * 															[3] \f$F_xx\f$<br>
 * 															[4] \f$F_xy\f$<br>
 * 															[5] \f$F_xz\f$<br>
 * 															[6] \f$F_yx\f$<br>
 * 															[7] \f$F_yy\f$<br>
 * 															[8] \f$F_yz\f$<br>
 * 															[9] \f$F_zx\f$<br>
 * 															[10] \f$F_zy\f$<br>
 * 															[11] \f$F_zz\f$<br>
 */
template<unsigned int spacedim>
class
InterfaceDissipation02:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Asymmetry factor
	 */
	const double
	alpha;

	/**
	 * gas constant time temperature
	 */
	const double
	RT;

	/**
	 * current density i0
	 */
	const double
	i0;

	/**
	 *
	 * If @p method==0: Miehe's method (first order accurate)
	 * If @p method==1: Miehe's method as prediction step + correction step (second order accurate)
	 */
	const unsigned int
	method;

	/**
	 * see ScalarFunctional::get_h_sigma
	 */
	bool
	get_h_sigma(const dealii::Vector<double>& 				e_sigma,
				const std::vector<dealii::Vector<double>>&	e_sigma_ref_sets,
				dealii::Vector<double>& 					hidden_vars,
				const dealii::Point<spacedim>& 				x,
				const dealii::Tensor<1,spacedim>& 			n,
				double& 									h_sigma,
				dealii::Vector<double>& 					h_sigma_1,
				dealii::FullMatrix<double>& 				h_sigma_2,
				const std::tuple<bool, bool, bool>			requested_quantities)
	const ;

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
	 * @param[in]		global_data				InterfaceDissipation02::global_data
	 *
	 * @param[in]		alpha					InterfaceDissipation02::alpha
	 *
	 * @param[in]		i0						InterfaceDissipation02::i0
	 *
	 * @param[in]		method					InterfaceDissipation02::method
	 */
	InterfaceDissipation02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
							const std::set<dealii::types::material_id>										domain_of_integration,
							const dealii::Quadrature<spacedim-1>											quadrature,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const double																	alpha,
							const double																	RT,
							const double																	i0,
							const unsigned int																method);
};

}



#endif /* INCREMENTALFE_SCALARFUNCTIONALS_INTERFACEDISSIPATION02_H_ */
