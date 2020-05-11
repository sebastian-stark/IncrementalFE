#ifndef INCREMENTALFE_SCALARFUNCTIONALS_SOLUTIONGELINTERACTION00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_SOLUTIONGELINTERACTION00_H_

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
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0]  \f$t_x\f$
 * 															[1]  \f$t_y\f$<br>
 * 															[2]  \f$t_z\f$<br>
 * 															[3]  \f$u^\mathrm{S(g)}_x\f$<br>
 * 															[4]  \f$u^\mathrm{S(g)}_y\f$<br>
 * 															[5]  \f$u^\mathrm{S(g)}_z\f$<br>
 * 															[6]  \f$I^\mathrm{F(g)}_n\f$<br>
 * 															[7]  \f$\eta^\mathrm{F(g)}\f$<br>
 * 															[8]  \f$u^\mathrm{F(s)}_x\f$<br>
 * 															[9]  \f$u^\mathrm{F(s)}_y\f$<br>
 * 															[10] \f$u^\mathrm{F(s)}_z\f$<br>
 * 															[11] \f$F^\mathrm{(g)}_xx\f$<br>
 * 															[12] \f$F^\mathrm{(g)}_xy\f$<br>
 * 															[13] \f$F^\mathrm{(g)}_xz\f$<br>
 * 															[14] \f$F^\mathrm{(g)}_yx\f$<br>
 * 															[15] \f$F^\mathrm{(g)}_yy\f$<br>
 * 															[16] \f$F^\mathrm{(g)}_yz\f$<br>
 * 															[17] \f$F^\mathrm{(g)}_zx\f$<br>
 * 															[18] \f$F^\mathrm{(g)}_zy\f$<br>
 * 															[19] \f$F^\mathrm{(g)}_zz\f$<br>
 */
template<unsigned int spacedim>
class
SolutionGelInteraction00:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * molar volume of fluid (m^3/mol)
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
	 * @param[in]		global_data				SolutionGelInteraction00::global_data
	 *
	 * @param[in]		V_m_F					SolutionGelInteraction00::V_m_F
	 *
	 * @param[in]		method					SolutionGelInteraction00::method
	 */
	SolutionGelInteraction00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
								const std::set<dealii::types::material_id>										domain_of_integration,
								const dealii::Quadrature<spacedim-1>											quadrature,
								GlobalDataIncrementalFE<spacedim>&												global_data,
								const double																	V_m_F,
								const unsigned int 																method);
};

}



#endif /* INCREMENTALFE_SCALARFUNCTIONALS_DUALINTERFACEDISSIPATION02_H_ */
