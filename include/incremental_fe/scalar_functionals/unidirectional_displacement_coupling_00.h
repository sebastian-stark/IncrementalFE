#ifndef INCREMENTALFE_SCALARFUNCTIONALS_UNIDIRECTIONALDISPLACEMENTCOUPLING00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_UNIDIRECTIONALDISPLACEMENTCOUPLING00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining unidirectional displacement coupling, where a domain (s) is coupled to a
 * domain (g) and no forces are exerted on (g) due to the coupling.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma:<br>	[0] \f$t_x\f$<br>
 * 															[1] \f$t_y\f$<br>
 * 															[2] \f$t_z\f$<br>
 * 															[3]	\f$u^{(s)}_x\f$<br>
 * 															[4]	\f$u^{(s)}_y\f$<br>
 * 															[5]	\f$u^{(s)}_z\f$<br>
 * 															[6]	\f$u^{(g)}_x\f$<br>
 * 															[7]	\f$u^{(g)}_y\f$<br>
 * 															[8]	\f$u^{(g)}_z\f$<br>
 */
template<unsigned int spacedim>
class UnidirectionalDisplacementCoupling00:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1 (see UnidirectionalDisplacementCoupling00::method).
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

	/**
	 * Function computing the value of the continuous potential and its derivatives
	 *
	 * @param[in]	e_sigma					The values of the dependent variables
	 *
	 * @param[out]	val						The value of the potential
	 *
	 * @param[in]	n						normal vector
	 *
	 * @param[out]	d1						First derivatives
	 *
	 * @param[out]	d2						Second derivatives
	 *
	 *
	 * @param[in]	requested_quantities	Indicates which quantities are to be computed
	 */
	void
	compute_derivatives(const dealii::Vector<double>&		e_sigma,
						const dealii::Tensor<1,spacedim>& 	n,
						double&								val,
						dealii::Vector<double>&				d1,
						dealii::FullMatrix<double>& 		d2,
						const std::tuple<bool, bool, bool>&	requested_quantities)
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
	 * @param[in]		global_data				UnidirectionalDisplacementCoupling00::global_data
	 *
	 * @param[in]		alpha					UnidirectionalDisplacementCoupling00::alpha
	 */
	UnidirectionalDisplacementCoupling00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
											const std::set<dealii::types::material_id>										domain_of_integration,
											const dealii::Quadrature<spacedim-1>											quadrature,
											GlobalDataIncrementalFE<spacedim>&												global_data,
											const double																	alpha);
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_UNIDIRECTIONALDISPLACEMENTCOUPLING00_H_ */
