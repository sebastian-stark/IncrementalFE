#ifndef INCREMENTALFE_SCALARFUNCTIONALS_ELECTRONEUTRALITYSURFACE00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_ELECTRONEUTRALITYSURFACE00_H_

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
 * \f$h^\Sigma_\tau = \varphi \boldsymbol{D} \cdot \boldsymbol{n}\f$,
 *
 * where \f$\boldsymbol{D}\f$ is a flux vector, \f$\boldsymbol{n}\f$ the
 * outward unit normal on the interface, and \f$\varphi\f$ a Lagrangian multiplier.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma:<br>	[0] \f$D_x\f$<br>
 * 															[1] \f$D_y\f$<br>
 * 															[2] \f$D_z\f$<br>
 * 															[3]	\f$\varphi\f$
 */
template<unsigned int spacedim>
class ElectroNeutralitySurface00:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1 (see ElectroNeutralitySurface00::method).
	 */
	const double
	alpha;

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
	 * @param[in]		global_data				ElectroNeutralitySurface00::global_data
	 *
	 * @param[in]		alpha					ElectroNeutralitySurface00::alpha
	 */
	ElectroNeutralitySurface00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						const double																	alpha);
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_ELECTRONEUTRALITYSURFACE00_H_ */
