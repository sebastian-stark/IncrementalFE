#ifndef INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL01_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL01_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining chemical potential associated with a single species in large deformation setting.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$
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
ChemicalPotential01:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
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
	 * Reference concentration (measured w.r.t. deformed volume) at which chemical potential is \f$RT\mu_0\f$
	 */
	const double
	c0;

	/**
	 * \f$\mu_0\f$
	 */
	const double
	mu0;

	/**
	 * Numerical parameter between @p 0 and @p 1 (weights derivatives at the beginning and at the end of the load step).
	 */
	const double
	alpha;

	/**
	 * Threshold concentration \f$c_\mathrm{th}/c_0\f$.
	 * The chemical potential expression tends to
	 * cause numerical difficulties for very low concentrations - therefore
	 * a logarithmic penalty type potential
	 * \f$h^\Omega_\rho = A \ln\dfrac{c}{J c_0} + B + RT c (\mu_0 - \ln J)\f$ is used for very small
	 * concentrations \f$c<c_\mathrm{th}\f$ in order to reduce
	 * these difficulties. The constants \f$A\f$ and \f$B\f$ are determined from
	 * the requirement of continuity for \f$h^\Omega_\rho\f$ and its first derivative
	 * at the concentration \f$c=c_\mathrm{th}\f$.
	 */
	const double
	c_th_c0;

	/**
	 * log(c_th_c0) (store this because its calculation is usually relatively expensive and
	 * it is required during every call of the routine
	 */
	const double
	log_c_th_c0;

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
	 * @param[in]		global_data				ChemicalPotential01::global_data
	 *
	 * @param[in]		RT						ChemicalPotential01::RT
	 *
	 * @param[in]		c0						ChemicalPotential01::c0
	 *
	 * @param[in]		mu0						ChemicalPotential01::mu0
	 *
	 * @param[in]		alpha					ChemicalPotential01::alpha
	 *
	 * @param[in]		c_th_c0					ChemicalPotential01::c_th_c0
	 */
	ChemicalPotential01(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																RT,
						const double																c0,
						const double																mu0,
						const double																alpha,
						const double																c_th_c0 = 0.0);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL01_H_ */
