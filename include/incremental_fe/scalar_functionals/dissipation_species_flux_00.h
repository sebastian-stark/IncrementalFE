#ifndef INCREMENTALFE_SCALARFUNCTIONALS_DISSIPATIONSPECIESFLUX00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_DISSIPATIONSPECIESFLUX00_H_

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
 * (1) \f$h^\Omega_\rho =	\dfrac{\Delta t}{2 D ( c^{(\alpha)} + c_0 ) }
 * 							\dfrac{\boldsymbol{I}-\boldsymbol{I}_\mathrm{ref}}{\Delta t} \cdot
 * 							\dfrac{\boldsymbol{I}-\boldsymbol{I}_\mathrm{ref}}{\Delta t}\f$
 *
 * or
 *
 * (2) \f$h^\Omega_\rho =	\dfrac{\Delta t}{2 D ( c^{(\alpha)} + c_0 ) }
 * 							\dfrac{\boldsymbol{I}}{\Delta t} \cdot
 * 							\dfrac{\boldsymbol{I}}{\Delta t}\f$.
 *
 * In these equations \f$c^{(\alpha)} = (1-\alpha)c_\mathrm{ref} + \alpha c\f$ is a species concentration, where the
 * parameter \f$\alpha\f$ determines at which point within the time step the species concentration is evaluated
 * (\f$c_\mathrm{ref}\f$ is the species concentration in the beginning of the time step, and \f$c\f$ is the species
 *  concentration at the end of the time step).
 *
 * \f$\Delta t\f$ is the length of the time step.
 *
 * The quantity \f$\boldsymbol{I}\f$ an accumulated flux vector.
 * This vector may be either the total flux from the beginning of the simulation
 * until the end of the current time step (in this case, the formulation (1) is appropriate, with
 * \f$\boldsymbol{I}_\mathrm{ref}\f$ being the total flux until the beginning of the current time step),
 * or it may be the total flux within the time step (in this case, the formulation (2) is appropriate).
 *
 * The "mobility" \f$D\f$ is related to the usual "diffusion constant" \f$ \bar D \f$ by \f$D = \bar D/(RT)\f$.
 * Moreover, it is related to the electrical mobility \f$\mu\f$ by \f$ D = \mu n/F\f$, where \f$n\f$ is the charge per
 * molecule of the mobile species in multiples of the elementary charge and \f$F\f$ Faraday's constant.
 * The material parameter \f$c_0\f$ represents an "offset" for the concentration.
 *
 * There are two implementations available. With the first method, the potentials stated above are used as they are.
 * With the second method, however, the first derivatives of \f$h^\Omega_\rho\f$ are computed while regarding
 * \f$c^{(\alpha)}\f$ a fixed quantity (i.e., the derivative of \f$h^\Omega_\rho\f$ w.r.t. \f$c\f$ is set to zero).
 * While the first method has the advantage of producing symmetric second derivatives, the second method typically
 * yields more accurate results. However, for the special case of \f$\alpha = 0\f$ both methods are equivalent.
 *
 * The potential implemented by this class can be used to model the dissipation associated with the flux of a mobile species.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0] \f$I_x\f$<br>
 * 																				[1] \f$I_y\f$<br>
 * 																				[2] \f$I_z\f$<br>
 * 																				[3] \f$c\f$
 */
template<unsigned int spacedim>
class
DissipationSpeciesFlux00:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
{

private:

	/**
	 * global data object
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * Type of formulation.
	 *
	 * If @p formulation==0: in solution increments \f$\boldsymbol{I}-\boldsymbol{I}_\mathrm{ref}\f$,
	 * If @p formulation==1: in solution \f$\boldsymbol{I}\f$
	 */
	const unsigned int
	formulation;

	/**
	 * Diffusion coefficient of species \f$D\f$
	 */
	const double
	D;

	/**
	 * Offset concentration \f$c_0\f$
	 */
	const double
	c0;

	/**
	 *
	 * If @p method==0: implementation based on time discrete potential
	 * If @p method==1: first derivative of the integrand w.r.t. \f$c\f$ is
	 *    				set to zero
	 * If @p method==2: predictor - corrector mode based on time discrete potential
	 */
	const unsigned int
	method;

	/**
	 * Specifies how to treat second derivatives if DissipationSpeciesFlux00::method==1
	 * If @p 0: use exact (unsymmetric) second derivatives
	 * If @p 1: ignore unsymmetric contributions (i.e., all second derivatives involving
	 *    		\f$c\f$ are set to zero)
	 */
	const unsigned int
	sym_mode;

	/**
	 * Parameter \f$\alpha\f$ (between @p 0 and @p 1). In general, @p alpha=1.0 is used
	 * in the very first time step (to stay consistent with the implementation
	 * of other potentials, which do the same in order to handle cases,
	 * where the initial solution is not equilibrated)
	 */
	double
	alpha;

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
	 * @param[in]		global_data				DissipationSpeciedFlux01::global_data
	 *
	 * @param[in]		D						DissipationSpeciedFlux01::D
	 *
	 * @param[in]		c0						DissipationSpeciedFlux01::c0
	 *
	 * @param[in]		alpha					DissipationSpeciesFlux00::alpha
	 *
	 * @param[in]		method					DissipationSpeciesFlux00::method
	 *
	 * @param[in]		sym_mode				DissipationSpeciesFlux00::sym_mode
	 *
	 * @param[in]		formulation				DissipationSpeciesFlux00::formulation
	 */
	DissipationSpeciesFlux00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																D,
								const double																c0,
								const double																alpha,
								const unsigned int															method,
								const unsigned int															sym_mode,
								const unsigned int 															formulation = 0);

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_h_omega
	 */
	bool
	get_h_omega(	const dealii::Vector<double>&				e_omega,
					const std::vector<dealii::Vector<double>>&	e_omega_ref_sets,
					dealii::Vector<double>&						hidden_vars,
					const dealii::Point<spacedim>&				x,
					double&										h_omega,
					dealii::Vector<double>&						h_omega_1,
					dealii::FullMatrix<double>&					h_omega_2,
					const std::tuple<bool, bool, bool>			requested_quantities)
	const;

};

}




#endif /* INCREMENTALFE_SCALARFUNCTIONALS_DISSIPATIONSPECIESFLUX00_H_ */
