#ifndef INCREMENTALFE_SCALARFUNCTIONALS_INTERFACEDISSIPATION00_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_INTERFACEDISSIPATION00_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 *
 * Class defining an interface related scalar functional with the integrand
 *
 * (1) \f$h^\Sigma_\tau = \dfrac{1}{2 d} \dfrac{(I_\mathrm{n}-I_\mathrm{n,ref})^2}{\Delta t}\f$,
 *
 * or
 *
 * (2) \f$h^\Sigma_\tau = \dfrac{1}{2 d} \dfrac{I_\mathrm{n}^2}{\Delta t}\f$.
 *
 * In these equations \f$\Delta t\f$ is the length of the time step.
 *
 * The quantity \f$I_\mathrm{n}\f$ an accumulated flux vector through the interface.
 * This vector may be either the total flux from the beginning of the simulation
 * until the end of the current time step (in this case, the formulation (1) is appropriate, with
 * \f$I_\mathrm{n,ref}\f$ being the total flux until the beginning of the current time step),
 * or it may be the total flux within the time step (in this case, the formulation (2) is appropriate).
 *
 * The quantity \f$d\f$ is a "conductance" type material parameter.
 * *
 * The potential implemented by this class can be used to model the dissipation associated with the flux of a mobile species
 * through an interface.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$I_x\f$<br>
 * 															[1] \f$I_y\f$<br>
 * 															[2] \f$I_z\f$<br>
 */
template<unsigned int spacedim>
class
InterfaceDissipation00:public dealii::GalerkinTools::ScalarFunctional<spacedim-1, spacedim>
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
	 * If @p formulation==0: in solution increments \f$I_\mathrm{n}-I_\mathrm{n,ref}\f$,
	 * If @p formulation==1: in solution \f$I_\mathrm{n}\f$
	 */
	const unsigned int
	formulation;

	/**
	 * Interface conductance
	 */
	const double
	d;

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
	const ;

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
	 * @param[in]		global_data				InterfaceDissipation00::global_data
	 *
	 * @param[in]		d						InterfaceDissipation00::d
	 *
	 * @param[in]		formulation				InterfaceDissipation00::formulation
	 */
	InterfaceDissipation00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
							const std::set<dealii::types::material_id>										domain_of_integration,
							const dealii::Quadrature<spacedim-1>											quadrature,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const double																	d,
							const unsigned int																formulation = 0);
};

}



#endif /* INCREMENTALFE_SCALARFUNCTIONALS_INTERFACEDISSIPATION00_H_ */
