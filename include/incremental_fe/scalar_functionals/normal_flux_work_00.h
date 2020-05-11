#ifndef INCREMENTALFE_SCALARFUNCTIONALS_NORMALFLUXWORK00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_NORMALFLUXWORK00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$h^\Sigma_\tau = \tilde\varphi \boldsymbol{I} \cdot \boldsymbol{n}\f$,
 *
 * where \f$\boldsymbol{I}\f$ is a flux vector vector, \f$\boldsymbol{n}\f$ the
 * outward unit normal on the interface, and \f$\tilde\varphi\f$ the prescribed potential
 * (which may depend on time).
 *
 * Ordering of quantities in ScalarFunctional::e_sigma:<br>	[0] \f$I_x\f$<br>
 * 															[1] \f$I_y\f$<br>
 * 															[2] \f$I_z\f$
 */
template<unsigned int spacedim>
class NormalFluxWork00:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * %Function determining \f$\tilde\varphi(t)\f$
	 */
	dealii::Function<spacedim>&
	function_phi;

	/**
	 * Numerical parameter between @p 0 and @p 1 (see NormalFluxWork00::method).
	 *
	 * @warning In the very first time step @p alpha=1.0 is used irrespective of the
	 * 			value provided for @p alpha! This helps to avoid oscillations of the
	 * 			solution if the initial solution fields are not equilibrated.
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
	 * @param[in]		global_data				NormalFluxWork00::global_data
	 *
	 * @param[in] 		function_phi			NormalFluxWork00::function_phi
	 *
	 * @param[in]		alpha					NormalFluxWork00::alpha
	 */
	NormalFluxWork00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						dealii::Function<spacedim>& 													function_phi,
						const double																	alpha);
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_NORMALFLUXWORK00_H_ */
