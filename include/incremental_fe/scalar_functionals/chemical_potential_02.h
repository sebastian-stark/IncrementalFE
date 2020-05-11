#ifndef INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL02_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL02_H_

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/point.h>

#include <galerkin_tools/scalar_functional.h>
#include <incremental_fe/global_data_incremental_fe.h>

namespace incrementalFE
{

/**
 * Class defining chemical potential of a charged species moving in a fluid.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$<br>
 * 																					[1] \f$c_\mathrm{fluid}\f$
 */
template<unsigned int spacedim>
class
ChemicalPotential02:public dealii::GalerkinTools::ScalarFunctional<spacedim, spacedim>
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
	 * @param[in]		global_data				ChemicalPotential02::global_data
	 *
	 * @param[in]		RT						ChemicalPotential02::RT
	 *
	 * @param[in]		mu_0					ChemicalPotential02::mu_0
	 *
	 * @param[in]		alpha					ChemicalPotential02::alpha
	 */
	ChemicalPotential02(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																RT,
						const double																mu_0,
						const double																alpha);
};

}

#endif /* INCREMENTALFE_SCALARFUNCTIONALS_CHEMICALPOTENTIAL02_H_ */
