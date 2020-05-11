#ifndef INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL03_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL03_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining chemical potential of a charged species moving in a fluid w.r.t. a fictitious background solid skeleton,
 * where the motion of the solid skeleton is merely a parameter.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$<br>
 * 																					[1] \f$F_xx\f$<br>
 * 																					[2] \f$F_xy\f$<br>
 * 																					[3] \f$F_xz\f$<br>
 * 																					[4] \f$F_yx\f$<br>
 * 																					[5] \f$F_yy\f$<br>
 * 																					[6] \f$F_yz\f$<br>
 * 																					[7] \f$F_zx\f$<br>
 * 																					[8] \f$F_zy\f$<br>
 * 																					[9] \f$F_zz\f$<br>
 */
template<unsigned int spacedim>
class
ChemicalPotential03:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * gas constant times absolute temperature
	 */
	const double
	RT;

	/**
	 * \f$\mu_0\f$ (potential when \f$c = c_\mathrm{fluid}\f$
	 */
	const double
	mu_0;

	/**
	 * molar volume of fluid
	 */
	const double
	V_m_f;

	/**
	 * Numerical parameter between @p 0 and @p 1 (weights derivatives at the beginning and at the end of the load step).
	 */
	const double
	alpha;

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_h_omega
	 */
	bool
	get_h_omega(const dealii::Vector<double>& 				e_omega,
				const std::vector<dealii::Vector<double>>&	e_omega_ref_sets,
				dealii::Vector<double>&						hidden_vars,
				const dealii::Point<spacedim>&				x,
				double&										h_omega,
				dealii::Vector<double>&						h_omega_1,
				dealii::FullMatrix<double>&					h_omega_2,
				const std::tuple<bool, bool, bool>			requested_quantities)
	const;

	/**
	 * Function computing the value of the continuous potential and its derivatives
	 *
	 * @param[in]	e_omega					The values of the dependent variables
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
	compute_derivatives(const dealii::Vector<double>&		e_omega,
						double&								val,
						dealii::Vector<double>&				d1,
						dealii::FullMatrix<double>& 		d2,
						const std::tuple<bool, bool, bool>&	requested_quantities)
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
	 * @param[in]		global_data				ChemicalPotential03::global_data
	 *
	 * @param[in]		RT						ChemicalPotential03::RT
	 *
	 * @param[in]		mu_0					ChemicalPotential03::mu_0
	 *
	 * @param[in]		V_m_f					ChemicalPotential03::V_m_f
	 *
	 * @param[in]		alpha					ChemicalPotential03::alpha
	 */
	ChemicalPotential03(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																RT,
						const double																mu_0,
						const double																V_m_f,
						const double																alpha);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL03_H_ */
