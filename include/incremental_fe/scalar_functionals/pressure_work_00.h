#ifndef INCREMENTALFE_SCALARFUNCTIONALS_PRESSUREWORK00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_PRESSUREWORK00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 *
 * Class defining the mechanical interface interaction between a hydrogel and the external solution.
 * This couples the fluid velocity in the solution to the velocity of the gel surface, while taking the
 * fluid out/influx at this surface into account as well.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0]  \f$p\f$
 * 															[1]  \f$u_x\f$<br>
 * 															[2]  \f$u_y\f$<br>
 * 															[3]  \f$u_z\f$<br>
 * 															[4]  \f$F_xx\f$<br>
 * 															[5]  \f$F_xy\f$<br>
 * 															[6]  \f$F_xz\f$<br>
 * 															[7]  \f$F_yx\f$<br>
 * 															[8]  \f$F_yy\f$<br>
 * 															[9]  \f$F_yz\f$<br>
 * 															[10] \f$F_zx\f$<br>
 * 															[11] \f$F_zy\f$<br>
 * 															[12] \f$F_zz\f$<br>
 */
template<unsigned int spacedim>
class
PressureWork00:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Numerical parameter between @p 0 and @p 1 (weights derivatives at the beginning and at the end of the load step).
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
				const std::tuple<bool, bool, bool>			requested_quantities)
	const;

	/**
	 * Function computing the value of the continuous potential and its derivatives
	 *
	 * @param[in]	e_sigma					The values of the dependent variables
	 *
	 * @param[in]	n						Normal vector
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
	 * @param[in]		global_data				PressureWork00::global_data
	 *
	 * @param[in]		alpha					PressureWork00::alpha
	 */
	PressureWork00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
					const std::set<dealii::types::material_id>										domain_of_integration,
					const dealii::Quadrature<spacedim-1>											quadrature,
					GlobalDataIncrementalFE<spacedim>&												global_data,
					const double																	alpha);
};

}



#endif /* INCREMENTALFE_SCALARFUNCTIONALS_PRESSUREWORK00_H_ */
