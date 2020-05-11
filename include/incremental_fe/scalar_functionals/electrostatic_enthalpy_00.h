#ifndef INCREMENTALFE_SCALARFUNCTIONALS_ELECTROSTATICENTHALPY00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_ELECTROSTATICENTHALPY00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining large deformation electrostatic enthalpy
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$E_x\f$<br>
 * 																					[1] \f$E_y\f$<br>
 * 																					[2] \f$E_z\f$<br>
 * 																					[3] \f$F_xx\f$<br>
 * 																					[4] \f$F_xy\f$<br>
 * 																					[5] \f$F_xz\f$<br>
 * 																					[6] \f$F_yx\f$<br>
 * 																					[7] \f$F_yy\f$<br>
 * 																					[8] \f$F_yz\f$<br>
 * 																					[9] \f$F_zx\f$<br>
 * 																					[10] \f$F_zy\f$<br>
 * 																					[11] \f$F_zz\f$<br>
 */
template<unsigned int spacedim>
class
ElectrostaticEnthalpy00:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * permittivity
	 */
	const double
	kappa;

	/**
	 * true: omit Maxwell stress terms (attention: this results in an unsymmetric formulation!)
	 */
	const bool
	omit_maxwell_stress;

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
	 * @param[in]		global_data				ElectrostaticEnthalpy00::global_data
	 *
	 * @param[in]		kappa					ElectrostaticEnthalpy00::kappa
	 *
	 * @param[in]		omit_maxwell_stress		ElectrostaticEnthalpy00::omit_maxwell_stress
	 *
	 * @param[in]		alpha					ElectrostaticEnthalpy00::alpha
	 */
	ElectrostaticEnthalpy00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>							domain_of_integration,
							const dealii::Quadrature<spacedim>									quadrature,
							GlobalDataIncrementalFE<spacedim>&									global_data,
							const double														kappa,
							const bool															omit_maxwell_stress,
							const double														alpha);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_ELECTROSTATICENTHALPY00_H_ */
