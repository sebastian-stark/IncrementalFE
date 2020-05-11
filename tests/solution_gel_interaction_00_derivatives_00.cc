#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/solution_gel_interaction_00.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check()
{
	vector<DependentField<spacedim-1, spacedim>> dependent_fields(20, DependentField<spacedim-1, spacedim>("q"));

	const double V_m_F = 1.3;
	const unsigned int method = 0;

	Tensor<1,spacedim> n;
	if(spacedim == 3)
	{
		n[0] = 0.3;
		n[1] = -0.2;
		n[2] = sqrt(1.0 - n[0] * n[0] - n[1] * n[1]);
	}
	else
	{
		n[0] = 0.3;
		n[1] = -sqrt(1.0 - n[0] * n[0]);
	}

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(0.5);
	global_data.set_t(1.5);

	SolutionGelInteraction00<spacedim> dissipation_species_flux(	dependent_fields,
																	{},
																	QGauss<spacedim-1>(1),
																	global_data,
																	V_m_F,
																	method);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	const double N = e_omega.size();
	for(unsigned int m = 0; m < e_omega.size(); ++m)
	{
		e_omega[m] = - cos( 2.0*numbers::PI*((double)(m+1))/N ) - 1.1;
		e_omega_ref_sets[0][m] = - sin( 2.0*numbers::PI*((double)(m+1))/N ) - 1.1;
	}
	dissipation_species_flux.compare_derivatives_with_numerical_derivatives(e_omega,
																			e_omega_ref_sets,
																			hidden_vars,
																			x,
																			n);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();
}
