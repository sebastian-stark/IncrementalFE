#ifndef INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = RTc \left( \ln\dfrac{c}{c_0} + \mu_0 -1 \right)\f$,
 *
 * where \f$R\f$ is the gas constant, \f$T\f$ the temperature, \f$c_0\f$ a
 * reference species concentration, \f$\mu_0\f$ a corresponding reference value
 * for the potential, and \f$c\f$ the species concentration.
 *
 * This is a chemical potential of a single species (without any non-ideality and interactions)
 * for the isothermal case.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$
 */
template<unsigned int spacedim>
class
ChemicalPotential00:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
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
	 * Reference concentration at which chemical potential is \f$RT\mu_0\f$
	 */
	const double
	c0;

	/**
	 * \f$\mu_0\f$
	 */
	const double
	mu0;

	/**
	 * Numerical parameter between @p 0 and @p 1 (see ChemicalPotential00::method).
	 */
	const double
	alpha;

	/**
	 * Numerical method
	 * (@p 0:	take derivatives of scalar functional at time corresponding to @p alpha,
	 *  @p 1: 	weight derivatives at the beginning of the time step and at the end of
	 *  		the time step according to @p alpha)
	 */
	const unsigned int
	method;

	/**
	 * Threshold concentration \f$c_\mathrm{th}/c_0\f$.
	 * The chemical potential expression tends to
	 * cause numerical difficulties for very low concentrations - therefore
	 * a logarithmic penalty type potential
	 * \f$h^\Omega_\rho = A \ln\dfrac{c}{c_0} + B + RT \mu_0 c\f$ is used for very small
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
	 * @param[in]		global_data				ChemicalPotential00::global_data
	 *
	 * @param[in]		RT						ChemicalPotential00::RT
	 *
	 * @param[in]		c0						ChemicalPotential00::c0
	 *
	 * @param[in]		mu0						ChemicalPotential00::mu0
	 *
	 * @param[in]		alpha					ChemicalPotential00::alpha
	 *
	 * @param[in]		method					ChemicalPotential00::method
	 *
	 * @param[in]		c_th_c0					ChemicalPotential00::c_th_c0
	 */
	ChemicalPotential00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																RT,
						const double																c0,
						const double																mu0,
						const double																alpha,
						const unsigned int															method = 1,
						const double																c_th_c0 = 0.0);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL00_H_ */
